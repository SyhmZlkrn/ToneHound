"""Build/export a reviewed capture catalogue; no bulk crawl runs by default."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tonehound.catalogue import Catalogue
from tonehound.library_manager import fetch
from tonehound.pipeline import Pipeline,PipelineConfig
from tonehound.embed import EmbedConfig
from tonehound.index_probe import resolve_probe
from tonehound.tone3000 import Tone3000Client


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['status','build','export','import','trim'])
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--tone-ids',type=Path,help='Reviewed JSON array of TONE3000 tone IDs; downloads only when supplied.')
    parser.add_argument('--limit',type=int,default=500,help='Maximum total catalogue entries, not a quality guarantee.')
    parser.add_argument('--pack',type=Path,help='Descriptor pack path (*.tonelib).')
    parser.add_argument('--rebuild',action='store_true',help='Re-render all descriptors using the currently selected probe.')
    args=parser.parse_args();root=args.root.resolve();store=Catalogue(root)
    if not 1<=args.limit<=10000:parser.error('Limit must be 1–10000.')
    if args.action=='status':print(json.dumps(store.summary(),indent=2));return
    if args.action=='trim':print(json.dumps(store.prune(),indent=2));return
    if args.action in ('export','import'):
        if args.pack is None:parser.error('--pack is required.')
        if args.action=='export':store.export(args.pack)
        else:store.import_pack(args.pack)
    else:
        cfg=PipelineConfig(profile_dir=store.directory,cache_dir=root/'.cache',di_path=resolve_probe(root),
            embed=EmbedConfig(dtype='float16'),render_device='auto',persistent_catalogue=True,refresh_descriptors=args.rebuild)
        pipeline=Pipeline(cfg)
        progress=lambda stage,fraction,message:print(message,flush=True)
        if args.tone_ids:
            ids=json.loads(args.tone_ids.read_text(encoding='utf-8-sig'))
            if not isinstance(ids,list) or any(type(i) is not int or i<=0 for i in ids):parser.error('Tone IDs must be positive integers.')
            client=Tone3000Client(cache_dir=root/'.cache')
            for tone_id in dict.fromkeys(ids):
                if store.summary()['catalogue_count']>=args.limit:break
                fetch(root,client,{'tone_ids':[tone_id],'models_per_tone':1,'max_models':1,'managed_cache':True},progress)
                # Checkpoint and release old model files after every downloaded capture.
                pipeline._index=None
                pipeline.index(progress=progress)
                cfg.refresh_descriptors=False
        else:pipeline.index(progress=progress)
        if args.pack:store.export(args.pack)
    from tonehound.native_bridge import library
    library(root)
    print(json.dumps(store.summary(),indent=2))

if __name__=='__main__':main()
