#include "Bridge.h"
#if JUCE_WINDOWS
#include <windows.h>
#endif

juce::var readJson(const juce::File& file) { return juce::JSON::parse(file.loadFileAsString()); }
juce::var object(std::initializer_list<std::pair<juce::Identifier, juce::var>> values)
{
    auto* o = new juce::DynamicObject;
    for (auto& v : values) o->setProperty(v.first, v.second);
    return juce::var(o);
}
juce::File findProjectRoot()
{
    auto configured = juce::SystemStats::getEnvironmentVariable("TONEHOUND_ROOT", "");
    if (configured.isNotEmpty()) return juce::File(configured);
    auto path = juce::File::getSpecialLocation(juce::File::currentExecutableFile).getParentDirectory();
    for (int i = 0; i < 8; ++i, path = path.getParentDirectory())
        if (path.getChildFile("engine/tools/native_job.py").existsAsFile()) return path;
    return juce::File(TONEHOUND_PROJECT_ROOT);
}

#if JUCE_WINDOWS
// Windows job ownership makes Cancel/closing a plugin terminate *only this
// instance's* Python worker and children (yt-dlp/ffmpeg), including on a crash.
class OwnedProcess
{
public:
    ~OwnedProcess() { if (job) CloseHandle(job); if (process) CloseHandle(process); }
    bool start(const juce::StringArray& args, const juce::File& cwd)
    {
        juce::String command;
        for (const auto& arg : args) {
            command += "\"";
            int slashes = 0;
            for (auto c : arg) {
                if (c == '\\') { ++slashes; continue; }
                command += juce::String::repeatedString("\\", slashes * (c == '"' ? 2 : 1));
                slashes = 0;
                if (c == '"') command += "\\";
                command += juce::String::charToString(c);
            }
            command += juce::String::repeatedString("\\", slashes * 2) + "\" ";
        }
        std::wstring text(command.toWideCharPointer());
        STARTUPINFOW startup{}; startup.cb = sizeof(startup);
        PROCESS_INFORMATION info{};
        job = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!job || !SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) return false;
        if (!CreateProcessW(nullptr, text.data(), nullptr, nullptr, FALSE,
                            CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr,
                            cwd.getFullPathName().toWideCharPointer(), &startup, &info)) return false;
        process = info.hProcess;
        if (!AssignProcessToJobObject(job, process)) {
            TerminateProcess(process, 1); CloseHandle(info.hThread); return false;
        }
        ResumeThread(info.hThread); CloseHandle(info.hThread);
        return true;
    }
    bool running() const { return process && WaitForSingleObject(process, 0) == WAIT_TIMEOUT; }
    void kill() { if (job) TerminateJobObject(job, 1); }
private:
    HANDLE job = nullptr, process = nullptr;
};
#else
class OwnedProcess {
public:
    bool start(const juce::StringArray& args, const juce::File&) { return process.start(args, 0); }
    bool running() { return process.isRunning(); }
    void kill() { process.kill(); }
    ~OwnedProcess() { kill(); }
private: juce::ChildProcess process;
};
#endif

AnalysisBridge::AnalysisBridge(juce::File r) : juce::Thread("ToneHound analysis"), root(std::move(r)) { startThread(); }
AnalysisBridge::~AnalysisBridge() { signalThreadShouldExit(); cancelled = true; notify(); stopThread(5000); }
AnalysisBridge::Snapshot AnalysisBridge::snapshot() const { const juce::ScopedLock g(lock); return state; }
bool AnalysisBridge::submit(juce::var request)
{
    const juce::ScopedLock g(lock);
    if (state.busy) return false;
    pending = request; cancelled = false;
    state.busy = true; state.action = request["action"].toString();
    state.message = "Starting " + state.action + "..."; state.progress = -1;
    notify(); return true;
}
void AnalysisBridge::cancel() { cancelled = true; notify(); }
void AnalysisBridge::run()
{
    while (!threadShouldExit()) {
        juce::var request;
        { const juce::ScopedLock g(lock); request = pending; pending = juce::var{}; }
        if (request.isVoid()) { wait(100); continue; }
        const auto directory = root.getChildFile(".cache/native/jobs").getChildFile(juce::Uuid().toString());
        directory.createDirectory();
        const auto input = directory.getChildFile("request.json");
        const auto response = directory.getChildFile("response.json");
        const auto progress = directory.getChildFile("progress.json");
        input.replaceWithText(juce::JSON::toString(request));
        auto python = readJson(root.getChildFile(".cache/native/runtime.json"))["python"].toString();
        if (python.isEmpty()) python = "python.exe";
        OwnedProcess child;
        juce::var result;
        if (!root.getChildFile("engine/tools/native_job.py").existsAsFile())
            result = object({{"ok", false}, {"message", "Analysis engine not found. Set TONEHOUND_ROOT to your ToneHound project folder."}});
        else if (!child.start({python, root.getChildFile("engine/tools/native_job.py").getFullPathName(),
                               "--request", input.getFullPathName(), "--response", response.getFullPathName(),
                               "--progress", progress.getFullPathName()}, root))
            result = object({{"ok", false}, {"message", "Could not start the Python analysis engine. Check .cache/native/runtime.json."}});
        else {
            while (child.running() && !threadShouldExit() && !cancelled.load()) {
                auto update = readJson(progress);
                if (update.isObject()) {
                    const juce::ScopedLock g(lock);
                    state.message = update["message"].toString(); state.stage = update["stage"].toString();
                    state.progress = (double)update["fraction"];
                }
                wait(100);
            }
            if (cancelled || threadShouldExit()) {
                child.kill(); result = object({{"ok", false}, {"message", "Analysis cancelled."}, {"code", "cancelled"}});
            } else result = readJson(response);
            if (!result.isObject()) result = object({{"ok", false}, {"message", "Analysis engine stopped unexpectedly. See " + response.withFileExtension(".log").getFullPathName()}});
        }
        { const juce::ScopedLock g(lock);
          state.response = result; state.busy = false; ++state.completion;
          state.progress = (bool)result["ok"] ? 1.0 : 0.0;
          state.message = (bool)result["ok"] ? "Ready" : result["message"].toString(); }
    }
}
