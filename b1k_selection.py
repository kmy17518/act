"""Bounded worker metadata: mmap episode records, index arrays, and integer lookups.

The parent keeps its convenient Python metadata. Spawned workers reopen this
selection snapshot instead of receiving O(episodes) Python objects/array copies.
"""
from collections import OrderedDict
from collections.abc import Mapping, Sequence
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import numpy as np


class MappedArray:
    def __init__(self, path):
        self.path = str(path)
        self._array = None

    def array(self):
        if self._array is None:
            self._array = np.load(self.path, mmap_mode='r')
        return self._array

    def __getstate__(self):
        return dict(path=self.path, _array=None)

    def __array__(self, dtype=None, copy=None):
        result = np.asarray(self.array(), dtype=dtype)
        return result.copy() if copy else result

    def __len__(self):
        return len(self.array())

    def __getitem__(self, key):
        return self.array()[key]

    @property
    def shape(self): return self.array().shape

    @property
    def dtype(self): return self.array().dtype

    def sum(self, *args, **kwargs): return self.array().sum(*args, **kwargs)


class MappedPositions(MappedArray):
    def index(self, value):
        index=int(np.searchsorted(self.array(),value))
        if index == len(self) or int(self[index]) != value:
            raise ValueError(value)
        return index


class MappedLookup(Mapping):
    def __init__(self, keys, values):
        self.keys_array, self.values_array = MappedArray(keys), MappedArray(values)

    def __getitem__(self, key):
        index = int(np.searchsorted(self.keys_array, int(key)))
        if index == len(self.keys_array) or int(self.keys_array[index]) != int(key):
            raise KeyError(key)
        return int(self.values_array[index])

    def __len__(self): return len(self.keys_array)

    def __iter__(self): return (int(x) for x in self.keys_array.array())


class MappedEpisodeLookup(Mapping):
    def __init__(self, positions, episodes):
        self.positions, self.episodes = positions, episodes

    def __getitem__(self, key): return self.episodes[self.positions[key]]
    def __iter__(self): return iter(self.positions)
    def __len__(self): return len(self.positions)


class MappedEpisodes(Sequence):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.offsets = MappedArray(self.directory/'episode_offsets.npy')
        self._bytes, self._cache = None, OrderedDict()

    def __getstate__(self):
        return dict(directory=self.directory, offsets=self.offsets, _bytes=None, _cache=OrderedDict())

    def __len__(self): return len(self.offsets)-1

    def __getitem__(self, index):
        if isinstance(index,slice): return [self[i] for i in range(*index.indices(len(self)))]
        index=int(index)
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        if index not in self._cache:
            if self._bytes is None:self._bytes=np.memmap(self.directory/'episodes.jsonbytes',dtype=np.uint8,mode='r')
            lo,hi=self.offsets[index:index+2]
            self._cache[index]=json.loads(self._bytes[lo:hi].tobytes())
            while len(self._cache)>16:self._cache.popitem(last=False)
        self._cache.move_to_end(index)
        return self._cache[index]


def compact_worker_state(state, directory, array_keys=(), lookup_keys=(), sampler_key=None):
    """Write once under a lock, then substitute pickle-small lazy descriptors."""
    arrays={key:np.asarray(state[key]) for key in array_keys if state.get(key) is not None}
    sampler_arrays={}
    if sampler_key:
        sampler_arrays={key:value for key,value in vars(state[sampler_key]).items() if isinstance(value,np.ndarray)}
    # Low-dimensional sidecars can be shared across horizons/stop controls, while
    # the sampler arrays and optional worker metadata cannot.
    groups={int(k):np.asarray(v,dtype=np.int64) for k,v in state.get('task_positions',{}).items()}
    digest=hashlib.sha256()
    digest.update(json.dumps(sorted(state['episodes'][0])).encode())
    for prefix,values in [('array',arrays),('sampler',sampler_arrays),('group',groups)]:
        for key,value in sorted(values.items()):
            digest.update(f'{prefix}:{key}:{value.dtype}:{value.shape}'.encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(json.dumps(list(lookup_keys)).encode())
    directory=Path(directory)/'worker-selection'/digest.hexdigest()[:24]
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        marker=directory/'complete.json'
        if not marker.is_file():
            offsets=[0]
            with (directory/'episodes.jsonbytes').open('wb') as f:
                for ep in state['episodes']:
                    f.write(json.dumps(ep,separators=(',',':')).encode());offsets.append(f.tell())
                f.flush();os.fsync(f.fileno())
            np.save(directory/'episode_offsets.npy',np.asarray(offsets,dtype=np.int64))
            for key,value in arrays.items():np.save(directory/(key+'.npy'),value)
            for key,value in sampler_arrays.items():np.save(directory/('sampler_'+key+'.npy'),value)
            for key,value in groups.items():np.save(directory/(f'group_{key}.npy'),value)
            for key in lookup_keys:
                items=sorted(state[key].items())
                np.save(directory/(key+'_keys.npy'),np.asarray([i[0] for i in items],dtype=np.int64))
                np.save(directory/(key+'_values.npy'),np.asarray([i[1] for i in items],dtype=np.int64))
            temporary=directory/'complete.tmp'
            temporary.write_text(json.dumps({'episodes':len(state['episodes']),'arrays':list(arrays),'sampler_arrays':list(sampler_arrays),'lookups':list(lookup_keys)}))
            os.replace(temporary,marker)
    result=state.copy()
    result['episodes']=MappedEpisodes(directory)
    for key in arrays:result[key]=MappedArray(directory/(key+'.npy'))
    for key in lookup_keys:result[key]=MappedLookup(directory/(key+'_keys.npy'),directory/(key+'_values.npy'))
    if groups:
        result['task_positions']={key:MappedPositions(directory/f'group_{key}.npy') for key in groups}
    if 'by_id' in state:
        result['by_id']=MappedEpisodeLookup(result['positions'],result['episodes'])
    if sampler_key:
        sampler=copy.copy(state[sampler_key])
        for key in sampler_arrays:setattr(sampler,key,MappedArray(directory/('sampler_'+key+'.npy')))
        result[sampler_key]=sampler
    return result
