"""Persistent descriptors with a bounded, reconstructible NAM working set.

Only downloads explicitly registered as managed may be evicted. Local imports
and pre-existing captures are preserved. SQLite commits each indexed model so
an interrupted catalogue build can resume without rendering everything again.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import sqlite3
import time
import zipfile
from contextlib import contextmanager

import numpy as np

from .tone_index import ToneEntry, ToneIndex, file_key, scan, _array_key, _read_cache
from .tone3000 import API, Model, Tone3000Client

DEFAULT_CACHE_BYTES = 32 * 1024 * 1024


class Catalogue:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.base = self.root / '.cache/tone3000'
        self.directory = self.base / 'profiles'
        self.base.mkdir(parents=True, exist_ok=True)
        self.database = self.base / 'catalogue.sqlite'
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS models (
                model_id INTEGER PRIMARY KEY, tone_id INTEGER NOT NULL,
                filename TEXT NOT NULL, name TEXT NOT NULL, file_key TEXT NOT NULL,
                sha256 TEXT NOT NULL, managed INTEGER NOT NULL DEFAULT 0,
                accessed REAL NOT NULL, embed_key TEXT, di_key TEXT, vector BLOB)''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def rows(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM models ORDER BY model_id')]

    def path(self, row):
        name = row['filename']
        if Path(name).name != name or re.search(r'[<>:"/\\|?*\x00-\x1f]',name) or not name.endswith('.nam'):
            raise ValueError('Invalid catalogue file name')
        path = self.directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError('Capture resolves outside its cache')
        return path

    def register(self, entry: ToneEntry, *, managed=False):
        if entry.model_id is None or entry.tone_id is None:
            return
        raw = entry.path.read_bytes()
        with self.connect() as db:
            old = db.execute('SELECT * FROM models WHERE model_id=?', (entry.model_id,)).fetchone()
            same = old is not None and old['sha256'] == hashlib.sha256(raw).hexdigest()
            db.execute('''INSERT OR REPLACE INTO models VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
                (entry.model_id, entry.tone_id, entry.path.name, entry.name, entry.key,
                 hashlib.sha256(raw).hexdigest(), int(bool(managed or (old and old['managed']))),
                 old['accessed'] if same else time.time(), old['embed_key'] if same else None,
                 old['di_key'] if same else None, old['vector'] if same else None))

    def ensure(self, filename: str, client=None) -> Path:
        # Resolve by stored identity, never a caller-provided download URL/path.
        with self.connect() as db:
            row = db.execute('SELECT * FROM models WHERE filename=?', (filename,)).fetchone()
            if row is None:
                raise FileNotFoundError('Capture is not in the catalogue. Add it from Library again.')
            row = dict(row)
            db.execute('UPDATE models SET accessed=? WHERE model_id=?', (time.time(), row['model_id']))
        path = self.path(row)
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']:
            return path
        client = client or Tone3000Client(cache_dir=self.root / '.cache')
        model = Model.from_json(client.request(f"{API}/models/{row['model_id']}"))
        if model.id != row['model_id'] or model.tone_id != row['tone_id']:
            raise ValueError('Downloaded model identity does not match the catalogue')
        from .library_manager import _validate_profile
        stage = self.base / 'incoming' / f'{os.getpid()}-{time.time_ns()}'
        downloaded = None
        try:
            downloaded = client.download_model(model, stage)
            _validate_profile(downloaded)
            if hashlib.sha256(downloaded.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('This remote capture changed. Add it again to refresh its descriptor.')
            self.directory.mkdir(parents=True, exist_ok=True)
            downloaded.replace(path)
        finally:
            if downloaded is not None:
                downloaded.unlink(missing_ok=True)
            if stage.is_dir() and not any(stage.iterdir()):
                stage.rmdir()
        return path

    def protected(self):
        result = set()
        for lease in (self.base / 'leases').glob('*.json'):
            try:
                if time.time() - lease.stat().st_mtime < 120:
                    result.add(Path(json.loads(lease.read_text(encoding='utf-8-sig'))['path']).name)
            except (OSError, ValueError, KeyError):
                continue
        return result

    def prune(self, budget=DEFAULT_CACHE_BYTES, *, keep=()):
        budget = max(0, int(budget))
        protected = self.protected() | set(keep)
        removed = []
        # Serialize eviction with other processes' eviction/access updates.
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            present = []
            for row in db.execute('SELECT * FROM models WHERE managed=1 ORDER BY accessed'):
                path = self.path(row)
                try:
                    present.append((row, path, path.stat().st_size))
                except FileNotFoundError:
                    continue
            size = sum(length for _, _, length in present)
            for row, path, length in present:
                if size <= budget:
                    break
                # Unindexed files and active/recently requested captures stay.
                if row['vector'] is None or path.name in protected or time.time()-row['accessed'] < 30:
                    continue
                if path.name in self.protected():
                    continue
                self.path(row)  # Recheck the resolved boundary immediately before removal.
                path.unlink(missing_ok=True)
                size -= length
                removed.append(row['model_id'])
        return {'removed': removed, 'managed_bytes': size, 'budget_bytes': budget,
                'protected_overage': max(0, size-budget)}

    def summary(self):
        rows = self.rows()
        return {'catalogue_count': len(rows), 'indexed_count': sum(r['vector'] is not None for r in rows),
                'cached_count': sum(self.path(r).is_file() for r in rows),
                'descriptor_bytes': sum(len(r['vector'] or b'') for r in rows),
                'cache_budget_bytes': DEFAULT_CACHE_BYTES}

    def index(self, cfg, embedder, di, progress=None):
        di_key = _array_key(np.asarray(di, dtype=np.float32)[:int(cfg.index_seconds*48000)])
        embed_key = embedder.cfg.key()
        # LoRA descriptors live in their own hash-keyed index. Keep the shipped
        # standard descriptors intact when switching models or exporting packs.
        store_descriptors = not getattr(embedder, 'trained', False)
        local = scan(cfg.profile_dir)
        from .community import entries as community_entries
        local = list({e.key:e for e in local + community_entries(cfg.cache_dir.parent)}.values())
        for entry in local:
            self.register(entry)
        stored = self.rows()
        by_model = {r['model_id']:r for r in stored}
        refresh = getattr(cfg, 'refresh_descriptors', False)
        entries = {e.model_id: e for e in local if e.model_id is not None}
        entries.update({r['model_id']: ToneEntry(r['file_key'], r['name'], self.path(r),
                        'tone3000', r['tone_id'], r['model_id'])
                        for r in stored if r['model_id'] not in entries})
        found = [e for e in local if e.model_id is None] + list(entries.values())
        cached = {} if refresh else _read_cache(cfg.index_cache, embed_key, di_key)
        for row in stored:
            # A shipped descriptor is immutable until an explicit rebuild. The
            # recipient's personal DI does not invalidate its reference probe.
            if store_descriptors and not refresh and row['vector'] is not None and row['embed_key']==embed_key:
                vector = np.frombuffer(row['vector'], dtype='<f4').copy()
                if vector.size and np.isfinite(vector).all():
                    cached[row['file_key']] = vector
        kept, vectors = [], []
        last_prune = time.monotonic()
        for i, entry in enumerate(found):
            if progress:
                progress('indexing', i/max(1,len(found)), f'Preparing descriptor {i+1}/{len(found)}: {entry.name}')
            if entry.key not in cached:
                if entry.model_id is not None:
                    self.ensure(entry.path.name)
                one = ToneIndex.build([entry], di, embedder=embedder, seconds=cfg.index_seconds,
                                      render_device=cfg.render_device, work_dir=cfg.cache_dir)
                cached[entry.key] = one.vectors[0]
            vector = np.asarray(cached[entry.key], dtype='<f4')
            kept.append(entry)
            vectors.append(vector)
            old = by_model.get(entry.model_id)
            if store_descriptors and entry.model_id is not None and (refresh or old is None or old['vector'] is None or old['embed_key']!=embed_key):
                with self.connect() as db:
                    db.execute('UPDATE models SET embed_key=?,di_key=?,vector=? WHERE model_id=?',
                               (embed_key,di_key,vector.tobytes(),entry.model_id))
            if time.monotonic()-last_prune>5:
                self.prune(keep=(entry.path.name,))
                last_prune=time.monotonic()
        if not kept:
            raise FileNotFoundError('Add captures in Library before matching.')
        self.prune()
        result = ToneIndex(kept,np.stack(vectors),embed_key=embed_key,di_key=di_key)
        # Local full rigs have no remote model ID; persist their base descriptor too.
        from .tone_index import _write_cache
        _write_cache(cfg.index_cache, result)
        return result

    def export(self, destination: Path):
        """Portable descriptor pack: no NAM audio/models, paths, tokens or signed URLs."""
        rows = [r for r in self.rows() if r['vector'] is not None]
        if not rows:
            raise ValueError('Index at least one capture before exporting descriptors.')
        signatures = {(r['embed_key'],r['di_key']) for r in rows}
        if len(signatures) != 1:
            raise ValueError('Rebuild the catalogue with one embedding configuration before export.')
        metadata = [{k:r[k] for k in ('model_id','tone_id','filename','name','file_key','sha256','embed_key','di_key')} for r in rows]
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = destination.with_name(f'.{destination.name}.{os.getpid()}.{time.time_ns()}.tmp')
        try:
            with staged.open('wb') as out:
                np.savez_compressed(out, metadata=json.dumps(metadata),
                                    vectors=np.stack([np.frombuffer(r['vector'],dtype='<f4') for r in rows]))
            staged.replace(destination)
        finally:
            staged.unlink(missing_ok=True)

    def import_pack(self, source: Path):
        # Inspect archive sizes and NPY headers before allocating arrays.
        with zipfile.ZipFile(source) as archive:
            if sorted(archive.namelist()) != ['metadata.npy', 'vectors.npy']:
                raise ValueError('Invalid descriptor pack members')
            for item in archive.infolist():
                if item.file_size > 192*1024*1024:
                    raise ValueError('Descriptor pack is too large')
                with archive.open(item) as member:
                    version = np.lib.format.read_magic(member)
                    if version not in ((1, 0), (2, 0)):
                        raise ValueError('Unsupported descriptor array format')
                    read_header = (np.lib.format.read_array_header_1_0 if version==(1, 0)
                                   else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = read_header(member)
                    if dtype.hasobject:
                        raise ValueError('Object arrays are not valid descriptors')
                    if item.filename == 'vectors.npy':
                        if len(shape)!=2 or not 1<=shape[0]<=10000 or not 1<=shape[1]<=32768 or dtype!=np.dtype('<f4'):
                            raise ValueError('Invalid descriptor dimensions or type')
                    elif shape!=() or dtype.kind not in ('U', 'S'):
                        raise ValueError('Invalid descriptor metadata')
                    if int(np.prod(shape, dtype=np.int64))*dtype.itemsize > item.file_size:
                        raise ValueError('Truncated descriptor array')
        with np.load(source, allow_pickle=False) as data:
            rows=json.loads(str(data['metadata']))
            vectors=np.asarray(data['vectors'],dtype='<f4')
        if not isinstance(rows,list) or len(rows)!=len(vectors) or not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors,axis=1)<1.e-12):
            raise ValueError('Invalid descriptor pack')
        identities, filenames, signatures = set(), set(), set()
        for row,vector in zip(rows,vectors):
            if not isinstance(row,dict) or any(not isinstance(row.get(k),str) or not row[k] or len(row[k])>4096
                    for k in ('filename','name','file_key','sha256','embed_key','di_key')):
                raise ValueError('Invalid descriptor metadata fields')
            self.path(row)
            if not all(type(row.get(k)) is int and 0<row[k]<2**63 for k in ('model_id','tone_id')):
                raise ValueError('Invalid descriptor identity')
            if not row['filename'].startswith(f"t3k-{row['tone_id']}-{row['model_id']} "):
                raise ValueError('Descriptor filename does not match its identity')
            if any(re.fullmatch(r'[0-9a-f]{'+str(n)+'}',row[k]) is None for k,n in (('file_key',16),('sha256',64))):
                raise ValueError('Invalid descriptor content hash')
            if row['model_id'] in identities or row['filename'] in filenames:
                raise ValueError('Duplicate descriptor identity')
            identities.add(row['model_id']);filenames.add(row['filename'])
            signatures.add((row['embed_key'],row['di_key']))
        if len(signatures)!=1:
            raise ValueError('Descriptor pack mixes embedding configurations or probes')
        with self.connect() as db:
            for row,vector in zip(rows,vectors):
                db.execute('INSERT OR IGNORE INTO models VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                    (row['model_id'],row['tone_id'],row['filename'],row['name'],row['file_key'],row['sha256'],1,
                     0,row['embed_key'],row['di_key'],vector.tobytes()))
        return self.summary()
