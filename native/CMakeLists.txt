cmake_minimum_required(VERSION 3.24)
project(ToneHoundNative VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

add_subdirectory(vendor/JUCE EXCLUDE_FROM_ALL)
add_subdirectory(vendor/signalsmith-linear EXCLUDE_FROM_ALL)
add_subdirectory(vendor/signalsmith-stretch EXCLUDE_FROM_ALL)
set(NAM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vendor/NeuralAmpModelerCore")
file(GLOB NAM_SOURCES CONFIGURE_DEPENDS "${NAM_ROOT}/NAM/*.cpp" "${NAM_ROOT}/NAM/*.c" "${NAM_ROOT}/NAM/*/*.cpp")
add_library(tonehound_nam STATIC ${NAM_SOURCES})
target_include_directories(tonehound_nam SYSTEM PUBLIC "${NAM_ROOT}" "${NAM_ROOT}/Dependencies/eigen" "${NAM_ROOT}/Dependencies/nlohmann" "${NAM_ROOT}/Dependencies/AudioDSPTools/dsp")
target_compile_definitions(tonehound_nam PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN NAM_DEFAULT_MAX_BUFFER_SIZE=4096)
if(NOT MSVC)
  target_compile_options(tonehound_nam PRIVATE -O3 -Wno-deprecated-declarations)
endif()

juce_add_plugin(ToneHound
  COMPANY_NAME "ToneHound" BUNDLE_ID "audio.tonehound.plugin"
  PLUGIN_MANUFACTURER_CODE Thnd PLUGIN_CODE TnHd
  FORMATS VST3 PRODUCT_NAME "ToneHound"
  IS_SYNTH FALSE NEEDS_MIDI_INPUT FALSE NEEDS_MIDI_OUTPUT FALSE
  IS_MIDI_EFFECT FALSE EDITOR_WANTS_KEYBOARD_FOCUS TRUE
  COPY_PLUGIN_AFTER_BUILD FALSE VST3_CATEGORIES Fx Distortion)
juce_generate_juce_header(ToneHound)
target_sources(ToneHound PRIVATE Source/PluginProcessor.cpp Source/PluginEditor.cpp Source/Bridge.cpp
  Source/MatchDSP.cpp Source/StudioPanels.cpp Source/PitchShift.cpp)
target_include_directories(ToneHound PUBLIC Source)
target_compile_definitions(ToneHound PUBLIC
  JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0 JUCE_ASIO=1
  JUCE_MODAL_LOOPS_PERMITTED=1
  TONEHOUND_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(ToneHound PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,tonehound_nam>" juce::juce_audio_utils juce::juce_dsp
  juce::juce_recommended_config_flags signalsmith-stretch)

juce_add_gui_app(ToneHound_Standalone PRODUCT_NAME "ToneHound" COMPANY_NAME "ToneHound")
target_sources(ToneHound_Standalone PRIVATE Source/Standalone.cpp)
target_link_libraries(ToneHound_Standalone PRIVATE ToneHound)

juce_add_console_app(ToneHound_Validate PRODUCT_NAME "ToneHound-Validate")
set_target_properties(ToneHound_Validate PROPERTIES OUTPUT_NAME "ToneHound-Validate")
target_sources(ToneHound_Validate PRIVATE Source/Validate.cpp Source/MatchDSPTests.cpp)
target_link_libraries(ToneHound_Validate PRIVATE ToneHound)

if(MINGW)
  foreach(target ToneHound_Standalone ToneHound_VST3 ToneHound_Validate)
    target_link_options(${target} PRIVATE -static-libgcc -static-libstdc++)
  endforeach()
endif()
