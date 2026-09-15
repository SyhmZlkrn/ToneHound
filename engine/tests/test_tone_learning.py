import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tonehound.tone_index import ToneEntry, ToneIndex
from tonehound.tone_learning import (LearnedToneIndex, apply_active, fit, metrics,
                                     save_arrays, unit, validate_splits, mark_failures)


def corpus(tmp_path):
    rng = np.random.default_rng(718)
    entries = []
    for i in range(12):
        path = tmp_path/f'{i}.nam';path.write_text(str(i))
        entries.append(ToneEntry(str(i),str(i),path))
    tone = rng.normal(size=(12,24))
    x,labels,splits,takes,views = [],[],[],[],[]
    for take,split in enumerate(['train','train','validation','test']):
        nuisance = rng.normal(size=24)*.4
        for label in range(12):
            for view in ['clean-0dB','mastered-0dB']:
                x.append(tone[label]+nuisance+rng.normal(size=24)*.03)
                labels.append(label);splits.append(split);takes.append(take);views.append(view)
    data = dict(vectors=unit(x),labels=np.array(labels),splits=np.array(splits),
                takes=np.array(takes),views=np.array(views))
    take_info = [dict(name=str(i),split=s) for i,s in enumerate(['train','train','validation','test'])]
    return entries,data,take_info


def test_test_audio_and_labels_do_not_train_or_select_projection(tmp_path):
    entries,data,takes = corpus(tmp_path)
    original,report = fit(data,entries,'encoder',takes)
    changed = {k:v.copy() for k,v in data.items()}
    test = data['splits']=='test'
    changed['vectors'][test] *= -1
    changed['labels'][test] = (changed['labels'][test]+1)%len(entries)
    candidate,new_report = fit(changed,entries,'encoder',takes)
    for field in ('mean','projection','gallery'):
        np.testing.assert_array_equal(original[field],candidate[field])
    assert report['strength']==new_report['strength']
    assert report['test']!=new_report['test']


def test_saved_inference_reproduces_evaluation_and_retains_untrained_capture(tmp_path):
    entries,data,takes = corpus(tmp_path)
    artifact,report = fit(data,entries,'encoder',takes)
    meta = json.loads(str(artifact['metadata']))
    meta['activation_passed'] = True  # Fixture need not outperform an already perfect baseline.
    artifact['metadata'] = json.dumps(meta)
    path = tmp_path/'active.npz';save_arrays(path,**artifact)
    base = ToneIndex(entries,artifact['gallery'],'encoder')
    restored = LearnedToneIndex(base,path)
    test = data['splits']=='test'
    scores = np.stack([restored.similarities(q) for q in data['vectors'][test]])
    assert metrics(scores,data['labels'][test])==report['test']['selected']['all']
    base.entries.append(ToneEntry('new','New capture',tmp_path/'new.nam'))
    base.vectors = np.vstack([base.vectors,data['vectors'][0]])
    expanded = LearnedToneIndex(base,path)
    assert expanded.learning['covered']==12 and len(expanded.similarities(data['vectors'][0]))==13


def test_stale_rejected_corrupt_and_nonfinite_artifacts_fall_back(tmp_path):
    entries,data,takes = corpus(tmp_path)
    artifact,_ = fit(data,entries,'encoder',takes)
    path = tmp_path/'active.npz';base=ToneIndex(entries,artifact['gallery'],'different')
    save_arrays(path,**artifact)
    assert apply_active(base,path) is base
    base.embed_key='encoder'
    artifact['mean'][0] = np.nan
    save_arrays(path,**artifact)
    assert apply_active(base,path) is base
    path.write_bytes(b'not a model')
    assert apply_active(base,path) is base


def test_decoded_duplicate_di_is_rejected_across_containers(tmp_path):
    signal = (np.sin(np.arange(20*1000)*.3)*3000).astype('int16')
    a=tmp_path/'one.wav';b=tmp_path/'two.flac'
    sf.write(a,signal,1000,subtype='PCM_16');sf.write(b,signal,1000,subtype='PCM_16')
    with pytest.raises(ValueError,match='Duplicate DI'):
        validate_splits([dict(path=a,split='train'),dict(path=b,split='test')])


def test_full_rig_manifest_excludes_heads_and_modified_weights(tmp_path):
    from tonehound.community import entries
    from tonehound.tone_learning import digest
    folder=tmp_path/'.cache/community';(folder/'profiles').mkdir(parents=True)
    rows=[]
    for name,rig in [('Amp-Cab-test.nam','full-rig'),('Head.nam','head-only'),('Bad-Cab-test.nam','full-rig')]:
        path=folder/'profiles'/name;path.write_text(name)
        rows.append(dict(filename=name,rig=rig,sha256=digest(path)))
    (folder/'profiles'/'Bad-Cab-test.nam').write_text('changed')
    (folder/'manifest.json').write_text(json.dumps(dict(entries=rows)))
    assert [e.path.name for e in entries(tmp_path)]==['Amp-Cab-test.nam']


def test_failed_separation_remains_a_miss_in_test_denominator():
    scores=np.eye(12,dtype=np.float32)
    queries=np.eye(12,dtype=np.float32);queries[0]=0
    result=metrics(mark_failures(scores,queries),np.arange(12))
    assert result['queries']==12 and result['separation_failures']==1
    assert result['top1']==result['top5']==result['top10']==result['mrr']==11/12
