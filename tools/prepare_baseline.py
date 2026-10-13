#!/usr/bin/env python3
"""Prepare original (not baseline z-scored) BiggerFish traces for MariIRQ.

Usage:
  python tools/prepare_baseline.py fit --input datasets/raw/closed_world --output data/processed/chrome_cw [--mapping FILE] [--indices FILE]
  python tools/prepare_baseline.py test --input datasets/raw/evaluation/cross_time_day1 --reference data/processed/chrome_cw --output data/processed/e0_tests/cross_time_day1
  python tools/prepare_baseline.py unknown --input datasets/raw/open_world --reference data/processed/chrome_cw --output data/processed/ow_unknown

The fixed scale is fitted to the CW TRAIN subset only. New datasets never re-fit their own scale.
"""
import argparse
import json
import pickle
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from sklearn.model_selection import train_test_split

L = 5000


def canon(x):
    s = str(x).strip()
    if s.startswith('b\''):
        s = s[2:-1]
    if '://' in s or '.' in s:
        if '://' not in s: s = 'https://' + s
        p = urlsplit(s)
        if p.netloc:
            host = p.netloc.lower().removeprefix('www.')
            path = p.path.rstrip('/')
            return host + path
    return s.strip('/').lower()


def separate_traces(trace):
    a = np.asarray(trace)
    if a.dtype == object:
        if isinstance(trace, (list,tuple)) and len(trace) and isinstance(trace[0], (list,tuple,np.ndarray)):
            return [np.asarray(t, dtype=np.float32).reshape(-1) for t in trace]
        raise ValueError('Trace dtype object: inspect raw PKL schema first')
    if a.ndim == 1: return [a.astype(np.float32)]
    if a.ndim == 2 and a.shape[-1] == 1: return [a[:,0].astype(np.float32)]
    if a.ndim == 2: return [v.astype(np.float32) for v in a]
    if a.ndim == 3 and a.shape[-1] == 1: return [v[:,0].astype(np.float32) for v in a]
    raise ValueError(f'Cannot recognize trace shape {a.shape}; must supply RAW 1-channel IRQ traces')


def load_raw(path):
    path = Path(path)
    files = sorted(path.rglob('*.pkl')) if path.is_dir() else [path]
    if path.is_dir() and not files:
        # Only original RAW arrays, not previous standardised preprocess.npz.
        files = sorted(p for p in path.glob('*.npz') if p.name not in ('preprocess.npz','split_indices.npz','train.npz','valid.npz','test.npz'))
    if not files: raise FileNotFoundError(f'No raw .pkl / .npz in {path}')
    traces, labels = [], []
    for f in files:
        if f.suffix == '.npz':
            if f.name == 'preprocess.npz':
                raise ValueError('preprocess.npz may contain standardised baseline arrays; supply ORIGINAL raw PKL/NPZ instead.')
            d = np.load(f,allow_pickle=True)
            k = 'x' if 'x' in d else 'X' if 'X' in d else None
            if k is None or 'y' not in d: raise ValueError(f'{f} must have raw x/X and y')
            arr = np.asarray(d[k]); ys = np.asarray(d['y'])
            if arr.ndim == 3 and arr.shape[-1] == 2:
                raise ValueError(f'{f}: already 2-channel MariIRQ; use existing prepared data rather than reconverting')
            if len(arr) != len(ys): raise ValueError(f'{f}: X/y count mismatch')
            for x,y in zip(arr,ys):
                items=separate_traces(x)
                traces.extend(items); labels.extend([str(y)] * len(items))
            continue
        if f.suffix != '.pkl': raise ValueError(f'Unsupported raw file: {f}')
        with f.open('rb') as h:
            while True:
                try: obj = pickle.load(h)
                except EOFError: break
                if isinstance(obj, dict):
                    t = obj.get('traces', obj.get('trace', obj.get('x', obj.get('X'))))
                    label = obj.get('domain', obj.get('label', obj.get('y')))
                elif isinstance(obj, (tuple,list)) and len(obj) >= 2:
                    t,label = obj[0],obj[1]
                else:
                    raise ValueError(f'{f}: unsupported PKL item type/schema {type(obj)}')
                if label is None: raise ValueError(f'{f}: missing site label')
                items=separate_traces(t)
                traces.extend(items); labels.extend([str(label)] * len(items))
    if len(traces) != len(labels): raise AssertionError('Raw X/y count mismatch')
    print(f'Loaded {len(traces)} traces, {len(set(labels))} distinct raw labels, {len(files)} files from {path}',flush=True)
    return traces, labels


def read_mapping(path, labels):
    if not path:
        keys=sorted(set(map(canon,labels)))
        mapping={k:i for i,k in enumerate(keys)}
        print('WARNING: mapping inferred from sorted raw labels; to reproduce baseline exactly pass --mapping baseline label_mapping.json')
    else:
        obj=json.loads(Path(path).read_text(encoding='utf-8'))
        if isinstance(obj, dict):
            for key in ('label_mapping','domain_to_label','domain_to_idx','site_to_idx','class_to_idx'):
                if key in obj and isinstance(obj[key],(dict,list)):
                    obj=obj[key];break
        if isinstance(obj,list): mapping={canon(s):i for i,s in enumerate(obj)}
        elif isinstance(obj,dict):
            if all(str(v).lstrip('-').isdigit() for v in obj.values()):
                mapping={canon(k):int(v) for k,v in obj.items()}
            elif all(str(k).lstrip('-').isdigit() for k in obj.keys()):
                mapping={canon(v):int(k) for k,v in obj.items()}
            else: raise ValueError('Unrecognized label_mapping.json: expected site->ID or ID->site')
        else: raise ValueError('Invalid mapping JSON')
    if len(set(mapping.values()))!=len(mapping) or sorted(mapping.values())!=list(range(len(mapping))):
        raise ValueError('Expected unique contiguous class IDs 0..K-1')
    if len(mapping)!=100: raise ValueError(f'CW mapping must have 100 sites; found {len(mapping)}')
    return mapping


def map_y(labels, mapping, aliases=None):
    aliases=aliases or {}
    mapped=[]; bad=set()
    for raw in labels:
        c=canon(raw)
        c=aliases.get(c,c)
        if c not in mapping: bad.add(str(raw));mapped.append(-1)
        else: mapped.append(mapping[c])
    if bad: raise ValueError('Labels NOT present in frozen 100-class mapping (never automatically re-number): '+repr(sorted(bad)[:25])+'; provide --aliases CSV: source_label,target_label')
    return np.asarray(mapped,dtype=np.int64)


def get_aliases(csv_file):
    if not csv_file:return {}
    import csv
    with Path(csv_file).open(encoding='utf-8-sig',newline='') as h:
        rows=csv.DictReader(h)
        return {canon(r['source_label']):canon(r['target_label']) for r in rows}


def index_splits(n,y,path,seed):
    if path:
        d=np.load(path)
        def pick(candidates):
            for c in candidates:
                if c in d: return np.asarray(d[c],dtype=np.int64)
            raise KeyError(f'{path}: need one of {candidates}; actual keys={d.files}')
        i=pick(('train_idx','train_indices','train'))
        v=pick(('valid_idx','val_idx','valid_indices','val_indices','valid','val'))
        t=pick(('test_idx','test_indices','test'))
        if any(x.ndim!=1 or x.dtype.kind not in 'iu' for x in (i,v,t)): raise ValueError('Split indices must be 1-D integer arrays')
        joined=np.concatenate([i,v,t])
        if len(joined)!=n or set(joined.tolist())!=set(range(n)):
            raise ValueError('Saved split_indices.npz does not match raw trace order/count; DO NOT silently regenerate a split')
        print('Using exact baseline split_indices.npz')
    else:
        all_ids=np.arange(n)
        i_v,t=train_test_split(all_ids,test_size=.1,random_state=seed,stratify=y)
        i,v=train_test_split(i_v,test_size=.1,random_state=seed,stratify=y[i_v])
        print('WARNING: baseline indices NOT supplied; new fixed stratified 81/9/10 split generated. Comparisons are not exact paired-sample comparisons.')
    return i,v,t


def fit_scaler(traces,indices):
    lo=float('inf');hi=float('-inf')
    for k in indices:
        a=np.asarray(traces[int(k)],dtype=np.float32)[:L]
        v=a[(a!=0)&np.isfinite(a)]
        if len(v):lo=min(lo,float(v.min()));hi=max(hi,float(v.max()))
    if not np.isfinite(lo) or not hi>lo: raise ValueError('Invalid RAW IRQ min/max; are inputs already normalized?')
    if 0 <= lo and hi <= 1.5:
        raise ValueError('Input appears pre-normalized (all IRQ <= 1.5); MariIRQ needs ORIGINAL raw IRQ samples, not baseline normalized arrays.')
    return {'raw_nonzero_min':lo,'raw_nonzero_max':hi,'scaled_min':0.0,'scaled_max':1500.0,'fitted_on':'CW train only','seq_len':L,'sample_interval':0.001}


def to_x(traces,scaler):
    n=len(traces)
    X=np.zeros((n,L,2),dtype=np.float32)
    t=np.arange(L,dtype=np.float32)*.001
    t[0]=1e-6
    lo=scaler['raw_nonzero_min'];hi=scaler['raw_nonzero_max']
    if not hi>lo: raise ValueError('Invalid scaler')
    count_clip=0
    for i,item in enumerate(traces):
        a=np.asarray(item,dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(a)):
            raise ValueError(f'trace {i} has NaN/inf')
        if len(a)>L:a=a[:L]
        k=len(a)
        X[i,:k,0]=t[:k]
        vals=a.copy(); nz=vals!=0
        vals[nz]=(vals[nz]-lo)/(hi-lo)*1500.
        count_clip+=int(np.count_nonzero((vals[nz]<0)|(vals[nz]>1500)))
        X[i,:k,1]=np.clip(vals,0,1500)
    print(f'X={X.shape}; out-of-range nonzero values clipped={count_clip}; scale frozen to CW TRAIN range',flush=True)
    return X


def save_npz(path,X,y):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,X=X,y=y)
    print(f'Saved {path} X={X.shape} y={y.shape} observed classes={len(set(y.tolist()))}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('fit','test','unknown'))
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--reference',type=Path,help='Prepared CW directory; required for test/unknown')
    p.add_argument('--mapping',type=Path,help='Original baseline label_mapping.json (recommended for fit)')
    p.add_argument('--indices',type=Path,help='Original baseline split_indices.npz (recommended for fit)')
    p.add_argument('--aliases',type=Path,help='CSV source_label,target_label for documented crosspage mappings')
    p.add_argument('--seed',type=int,default=2024)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    traces,labels=load_raw(a.input)
    if a.mode=='fit':
        mapping=read_mapping(a.mapping,labels)
        y=map_y(labels,mapping,get_aliases(a.aliases))
        if len(set(y.tolist()))!=100: raise ValueError('Fit dataset must contain traces for all 100 classes')
        tr,va,te=index_splits(len(y),y,a.indices,a.seed)
        scaler=fit_scaler(traces,tr)
        X=to_x(traces,scaler)
        for part,idx in (('train',tr),('valid',va),('test',te)):
            save_npz(a.output/(part+'.npz'),X[idx],y[idx])
        (a.output/'mapping.json').write_text(json.dumps(mapping,indent=2,ensure_ascii=False),encoding='utf-8')
        (a.output/'scaler.json').write_text(json.dumps(scaler,indent=2),encoding='utf-8')
        np.savez_compressed(a.output/'split_indices.npz',train_idx=tr,valid_idx=va,test_idx=te)
    else:
        if a.reference is None: p.error('--reference is required in test/unknown mode')
        mapping=json.loads((a.reference/'mapping.json').read_text(encoding='utf-8'))
        scaler=json.loads((a.reference/'scaler.json').read_text(encoding='utf-8'))
        if a.mode=='unknown':
            collisions=sorted(set(map(canon,labels)).intersection(mapping))
            if collisions: raise ValueError(f'OW contains {len(collisions)} known-site labels (examples: {collisions[:10]}); remove overlap first')
            y=np.full(len(traces),100,dtype=np.int64)
        else:y=map_y(labels,mapping,get_aliases(a.aliases))
        X=to_x(traces,scaler)
        save_npz(a.output/('unknown.npz' if a.mode=='unknown' else 'test.npz'),X,y)
        (a.output/'reference.json').write_text(json.dumps({'reference':str(a.reference.resolve()),'mode':a.mode,'source':str(a.input.resolve()),'samples':len(y)},indent=2),encoding='utf-8')

if __name__=='__main__':main()
