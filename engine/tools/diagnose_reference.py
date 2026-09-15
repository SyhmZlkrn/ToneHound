"""Compare mono/stereo separation and stem choice on an actual reference excerpt."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import torch
from tonehound import ingest
from tonehound.embed import EmbedConfig
from tonehound.pipeline import Pipeline, PipelineConfig
from tonehound.separate import isolate

ROOT = Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',action='append',required=True)
    ap.add_argument('--start',type=float,default=22)
    ap.add_argument('--end',type=float,default=60)
    ap.add_argument('--output',type=Path,default=ROOT/'.cache/reference_diagnostics')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    pipe=Pipeline(PipelineConfig(profile_dir=ROOT/'.cache/tone3000/profiles',cache_dir=ROOT/'.cache',
                                 embed=EmbedConfig(dtype='float16'),separator_autocast=True,render_device='auto'))
    index=pipe.index();reports=[]
    for number,spec in enumerate(args.source):
        source=ingest.load(spec,cache_dir=ROOT/'.cache/downloads',seconds=args.end-args.start,start_s=args.start,mono=False)
        directory=args.output/f'{number:02d}';directory.mkdir(exist_ok=True)
        x=source.samples;print(f'Analyzing {source.name}: {x.shape}',flush=True)
        sf.write(directory/'original-stereo.wav',x,48000,subtype='FLOAT')
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        old=isolate(x.mean(axis=1),separator=pipe.separator,tag=f'audit_old_{number}')
        random.seed(0); np.random.seed(0); torch.manual_seed(0)
        corrected=isolate(x,separator=pipe.separator,tag=f'audit_stereo_{number}',keep_stems=True,preserve_stereo=True)
        for name,audio in corrected.stems.items():sf.write(directory/f'{name}.wav',audio,48000,subtype='FLOAT')
        sf.write(directory/'old-mono-guitar.wav',old.samples,48000,subtype='FLOAT')
        guitar=corrected.stems.get('guitar',corrected.samples)
        other=corrected.stems.get('other',np.zeros_like(guitar))
        variants={'old_mono_guitar':old.samples,'stereo_guitar_folded':guitar.mean(axis=1),
                  'stereo_guitar':guitar,'stereo_other':other,'stereo_guitar_other':guitar+other,'stereo_mix':x}
        report={'name':source.name,'url':source.url,'source_path':str(source.path),'start':args.start,'end':args.end,
                'seed':0,'index_size':len(index),'embed_config':pipe.cfg.embed.__dict__,
                'channels':x.shape[1], 'channel_correlation':float(np.corrcoef(x.T)[0,1]) if x.shape[1]==2 else None,
                'stem_rms':{k:float(np.sqrt(np.mean(v**2))) for k,v in corrected.stems.items()},'variants':{}}
        for name,audio in variants.items():
            before=time.monotonic();query=pipe.embedder.embed(audio,48000)
            np.save(directory/f'{name}-embedding.npy',query)
            matches=index.search(query,top=len(index))
            report['variants'][name]={'rms':float(np.sqrt(np.mean(audio**2))),
                                      'matches':[m.to_json() for m in matches]}
            print(name, [m.name if hasattr(m,'name') else m.entry.name for m in matches[:3]],f'{time.monotonic()-before:.1f}s',flush=True)
            (directory/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        reports.append(report)
        (args.output/'report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print('DONE',flush=True)

if __name__=='__main__': main()
