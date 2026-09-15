"""Compare actual song crops using base and active retrieval; no guessed labels."""
from pathlib import Path
import json
import tempfile
import numpy as np

from tonehound.pipeline import Pipeline, PipelineConfig
from tonehound.embed import EmbedConfig
from tonehound.index_probe import resolve_probe
from tonehound.tone_learning import apply_active, save_arrays
from tonehound import ingest
from tonehound.separate import Separator, isolate
from tonehound.config import SAMPLE_RATE


def main():
    root=Path(__file__).resolve().parents[2]
    cfg=PipelineConfig(profile_dir=root/'.cache/tone3000/profiles',cache_dir=root/'.cache',
        di_path=resolve_probe(root),embed=EmbedConfig(dtype='float16'),render_device='auto',
        persistent_catalogue=True,learned_retrieval=False)
    pipeline=Pipeline(cfg);base=pipeline.index();active=apply_active(base,root/'.cache/tone_learning/active.npz')
    rows=[];vectors=[]
    with tempfile.TemporaryDirectory(prefix='tonehound_song_eval_') as work:
        separator=Separator(root/'.cache/uvr_models',work_dir=work,autocast=True,shifts=2)
        for song in sorted((root/'assets/song_train').glob('*.mp3')):
            audio=ingest.load(song,seconds=38,start_s=22,mono=False,cache_dir=root/'.cache/downloads')
            recovered=isolate(audio.samples,SAMPLE_RATE,separator=separator,preserve_stereo=True,tag='song')
            if recovered.stem not in ('guitar','other'):
                rows.append(dict(song=song.name,start_s=22,end_s=60,stem='failed',base=[],active=[]))
                vectors.append(np.zeros(base.dim,dtype=np.float32));continue
            query=pipeline.embedder.embed(recovered.samples,SAMPLE_RATE);vectors.append(query)
            def ranked(index):
                return [dict(name=m.entry.name,key=m.entry.key,score=m.similarity) for m in index.search(query,5)]
            rows.append(dict(song=song.name,start_s=22,end_s=60,stem=recovered.stem,
                             base=ranked(base),active=ranked(active)))
            print(song.name+' -> '+rows[-1]['active'][0]['name'],flush=True)
    report=dict(note='Unlabelled listening shortlist, not identification accuracy. These songs also supply backing stems for training/validation/testing.',
                retrieval=getattr(active,'learning',{}),songs=rows)
    output=root/'.cache/tone_learning'
    (output/'song_auditions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save_arrays(output/'song_queries.npz',vectors=np.asarray(vectors),names=np.asarray([r['song'] for r in rows]))


if __name__=='__main__':
    main()
