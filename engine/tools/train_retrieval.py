"""Reproducible multi-DI feature extraction, fitting and held-out evaluation."""
from pathlib import Path
import argparse
import json

from tonehound.embed import EmbedConfig, MertEmbedder
from tonehound.tone_index import scan
from tonehound.tone_learning import extract, fit, save_arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--profiles', type=Path, action='append')
    p.add_argument('--train-di', type=Path, action='append')
    p.add_argument('--validation-di', type=Path)
    p.add_argument('--test-di', type=Path)
    p.add_argument('--activate', action='store_true', help='Activate only if the fixed non-regression gate passes')
    p.add_argument('--with-song-separation', action='store_true', help='Include real song backing remixes passed through htdemucs_6s')
    args = p.parse_args()
    root = args.root.resolve()
    from tonehound.community import entries as community_entries
    found = ([e for folder in args.profiles for e in scan(folder)] if args.profiles else
             scan(root/'.cache/tone3000/profiles') + community_entries(root))
    entries = list({e.key:e for e in found}.values())
    if not entries:
        p.error('No NAM profiles found')
    di = root/'assets/user_di'
    trains = args.train_di or [di/'Djent DI.wav', di/'Funk DI.wav']
    takes = [dict(name=f.stem, path=f, split='train') for f in trains]
    takes += [dict(name=f.stem, path=f, split=s) for f,s in
              [(args.validation_di or di/'Thall DI.wav','validation'),
               (args.test_di or di/'Baritone DI.wav','test')]]
    cache = root/'.cache/tone_learning'
    embedder = MertEmbedder(EmbedConfig(dtype='float16'))
    data = extract(entries, takes, embedder, cache/'features', device='auto')
    save_arrays(cache/'dataset.npz', **data)
    separation = None
    if args.with_song_separation:
        from tonehound.separation_learning import extract_separated
        import numpy as np
        extra,separation = extract_separated(entries,takes,embedder,root)
        data = {k:np.concatenate([data[k],extra[k]]) for k in data}
        save_arrays(cache/'dataset_separated.npz', **data)
    artifact, report = fit(data, entries, embedder.cfg.key(), takes,
                           selection_condition='separated' if separation else None)
    if separation:
        from tonehound.tone_learning import LearnedToneIndex, metrics, mark_failures
        from tonehound.tone_index import ToneIndex
        import numpy as np
        report['conditions'] = 'Known NAM guitar + real song drum/bass/vocal stems, EQ/limiting, MP3, htdemucs_6s; DI and backing songs held out together'
        report['song_separation'] = separation
        if (cache/'active.npz').is_file():
            previous = LearnedToneIndex(ToneIndex(entries,np.zeros((len(entries),embedder.dim),np.float32),embedder.cfg.key()),cache/'active.npz')
            report['previous_active_test'] = {}
            for condition in ('clean','mastered','separated'):
                mask = (data['splits']=='test') & np.char.startswith(data['views'],condition)
                scores = np.stack([previous.similarities(q) for q in data['vectors'][mask]])
                report['previous_active_test'][condition] = metrics(mark_failures(scores,data['vectors'][mask]),data['labels'][mask])
            # Require the new separated metric to improve the old active metric,
            # with <= 2 percentage points clean top-5 loss. No retuning on test.
            new = report['test']['selected']; old = report['previous_active_test']
            report['activation_passed'] = bool(report['activation_passed'] and
                new['separated']['top5'] >= old['separated']['top5'] and
                new['separated']['mrr'] > old['separated']['mrr'] and
                new['clean']['top5'] >= old['clean']['top5']-.02)
        meta = json.loads(str(artifact['metadata']))
        meta.update(report=report,activation_passed=report['activation_passed'])
        artifact['metadata'] = json.dumps(meta)
    save_arrays(cache/'candidate.npz', **artifact)
    (cache/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if args.activate and report['activation_passed']:
        save_arrays(cache/'active.npz', **artifact)
    print(json.dumps(report, indent=2), flush=True)
    print('Active artifact written' if args.activate and report['activation_passed'] else 'Candidate saved; active artifact unchanged', flush=True)


if __name__ == '__main__':
    main()
