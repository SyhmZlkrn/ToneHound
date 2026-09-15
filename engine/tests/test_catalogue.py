import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from tonehound.catalogue import Catalogue
from tonehound.tone_index import ToneEntry, file_key, _array_key


def add(store, model, managed=True):
    path=store.directory/f't3k-10-{model} Test.nam'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'architecture':'Linear','config':{'receptive_field':1,'bias':False},'weights':[.5]}))
    entry=ToneEntry(file_key(path),path.stem,path,'tone3000',10,model)
    store.register(entry,managed=managed)
    return entry


def indexed(store, entry, vector, di_key='di'):
    with store.connect() as db:
        db.execute('UPDATE models SET vector=?,embed_key=?,di_key=?,accessed=? WHERE model_id=?',
            (np.asarray(vector,dtype='<f4').tobytes(),'embed',di_key,time.time()-3600,entry.model_id))


def test_eviction_retains_descriptor_and_protects_local_unindexed_and_active_files(tmp_path):
    store=Catalogue(tmp_path)
    entries=[add(store,i,managed=i!=3) for i in range(1,5)]
    for e in entries[:3]: indexed(store,e,[1,0,0])
    leases=store.base/'leases';leases.mkdir()
    (leases/'active.json').write_text(json.dumps({'path':str(entries[1].path)}))
    result=store.prune(0)
    assert result['removed']==[1]
    assert not entries[0].path.exists()
    assert all(e.path.exists() for e in entries[1:])
    assert store.summary()['indexed_count']==3


def test_500_descriptor_pack_searches_without_500_nam_files(tmp_path):
    source=Catalogue(tmp_path/'source')
    rng=np.random.default_rng(3)
    di=np.zeros(48000,dtype=np.float32)
    for i in range(500):
        entry=add(source,i+1)
        vector=rng.normal(size=4096).astype(np.float32);vector/=np.linalg.norm(vector)
        # Unique keys emulate distinct captures; the test's fixture NAM is shared.
        with source.connect() as db:
            db.execute('UPDATE models SET file_key=? WHERE model_id=?',(f'{i:016x}',i+1))
        indexed(source,entry,vector,_array_key(di))
    pack=tmp_path/'library.tonelib';source.export(pack)
    assert pack.stat().st_size<10*1024*1024
    target=Catalogue(tmp_path/'target');target.import_pack(pack)
    assert not list(target.directory.glob('*.nam'))
    cfg=SimpleNamespace(index_seconds=1,profile_dir=target.directory,index_cache=tmp_path/'unused.npz',
                        render_device='cpu',cache_dir=target.root/'.cache')
    embedder=SimpleNamespace(cfg=SimpleNamespace(key=lambda:'embed'))
    # A different user's probe must not force 500 remote captures to download.
    index=target.index(cfg,embedder,di+0.1)
    assert len(index)==500
    assert index.search(index.vectors[317],1)[0].entry.model_id==318
    assert not list(target.directory.glob('*.nam'))


def test_missing_capture_redownload_checks_identity_and_hash(tmp_path):
    store=Catalogue(tmp_path);entry=add(store,7);raw=entry.path.read_bytes();entry.path.unlink()
    class Client:
        def request(self,url):
            return {'id':7,'tone_id':10,'name':'Test','model_url':'https://example.invalid/test'}
        def download_model(self,model,directory):
            directory.mkdir(parents=True);p=directory/'test.nam';p.write_bytes(raw);return p
    assert store.ensure(entry.path.name,Client()).read_bytes()==raw
    with pytest.raises(FileNotFoundError):store.ensure('../outside.nam',Client())


def test_registration_preserves_existing_vectors_and_lru_time(tmp_path):
    store=Catalogue(tmp_path);entry=add(store,1);indexed(store,entry,[1,0,0])
    before=store.rows()[0]
    store.register(entry)
    after=store.rows()[0]
    assert after['vector']==before['vector'] and after['accessed']==before['accessed']


def test_concurrent_eviction_is_idempotent(tmp_path):
    store=Catalogue(tmp_path)
    for i in range(8):
        entry=add(store,i+1);indexed(store,entry,[1,0,0])
    with ThreadPoolExecutor(max_workers=2) as workers:
        reports=list(workers.map(lambda _:store.prune(0),range(2)))
    assert sum(len(r['removed']) for r in reports)==8
    assert store.summary()['cached_count']==0 and store.summary()['indexed_count']==8


@pytest.mark.parametrize('fault',['path','identity','duplicate','nonfinite','empty_vector','too_wide'])
def test_invalid_pack_is_rejected_without_partial_import(tmp_path,fault):
    source=Catalogue(tmp_path/'source');entry=add(source,1);indexed(source,entry,[1,0,0])
    row={k:v for k,v in source.rows()[0].items() if k not in ('vector','accessed','managed')}
    rows=[row];vectors=np.array([[1,0,0]],dtype='<f4')
    if fault=='path':row['filename']='t3k-10-1 Test:alternate-stream.nam'
    elif fault=='identity':row['model_id']=2
    elif fault=='duplicate':rows=[row,row];vectors=np.vstack([vectors,vectors])
    elif fault=='nonfinite':vectors[0,0]=np.nan
    elif fault=='empty_vector':vectors[:]=0
    elif fault=='too_wide':vectors=np.ones((1,32769),dtype='<f4')
    pack=tmp_path/'bad.tonelib'
    with pack.open('wb') as out:np.savez_compressed(out,metadata=json.dumps(rows),vectors=vectors)
    target=Catalogue(tmp_path/'target')
    with pytest.raises(ValueError):target.import_pack(pack)
    assert target.summary()['catalogue_count']==0
