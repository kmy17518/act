"""Shards, tolerance, portable incremental rebuild, and interrupted publication."""
import json
import os
import pickle
from pathlib import Path
import av
import numpy as np
import pytest
try:
    import b1k_cache_v2 as cache
except ImportError:
    from diffusion_policy.b1k import b1k_cache_v2 as cache


@pytest.fixture
def source(tmp_path):
    root=tmp_path/'data'; (root/'meta').mkdir(parents=True)
    (root/'meta/info.json').write_text('{}')
    path=root/'videos/head/chunk-000/file-000.mkv';path.parent.mkdir(parents=True)
    with av.open(str(path),'w') as c:
        stream=c.add_stream('ffv1',rate=10);stream.width=stream.height=32;stream.pix_fmt='bgr0'
        for i in range(10):
            pixels=np.full((32,32,3),i*20,np.uint8)
            pixels[:,:,1]=np.arange(32)[None,:]*4
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels,format='rgb24')):c.mux(packet)
        for packet in stream.encode():c.mux(packet)
    return root,path,tmp_path/'cache'


def test_shards_readback_limits_incremental_and_pickle(source):
    root,path,directory=source
    manifests=cache.build_group([path],directory,root,(32,32),shard_bytes=1600,batch_size=3)
    manifest=manifests[0];stem=cache.stem_for(directory,root,path)
    assert len(manifest['shards'])>1
    assert all((stem.parent/p).stat().st_size<=1600 for p in manifest['shards'])
    reader=cache.ShardedReader(directory,root,(32,32),max_entries=1,max_shards=1)
    np.testing.assert_array_equal(reader.positions(path,[0.,.3,.9]),[0,3,9])
    with av.open(str(path)) as c:
        for index,frame in enumerate(c.decode(video=0)):
            pixels=frame.to_ndarray(format='rgb24')
            actual=reader.read(path,index)
            np.testing.assert_array_equal(actual,cache.jpeg_roundtrip(pixels))
            assert np.square(actual.astype(float)-pixels).mean()<20
    restored=pickle.loads(pickle.dumps(reader))
    np.testing.assert_array_equal(restored.read(path,3),reader.read(path,3))
    assert len(reader.entries)==len(reader.shards)==1
    reader.close();restored.close()
    os.utime(path,None)
    assert cache.build_group([path],directory,root,(32,32),shard_bytes=1600)==[]


def test_rebuild_crash_keeps_old_entry(source,monkeypatch):
    root,path,directory=source
    cache.build_group([path],directory,root,(32,32))
    reader=cache.ShardedReader(directory,root,(32,32));expected=reader.read(path,4).copy();reader.close()
    stem=cache.stem_for(directory,root,path);old=json.loads(stem.with_suffix('.json').read_text())
    def crash(*args):raise RuntimeError('simulated crash before publication')
    monkeypatch.setattr(cache,'atomic_json',crash)
    with pytest.raises(RuntimeError):cache.build_group([path],directory,root,(32,32),quality=92)
    assert json.loads(stem.with_suffix('.json').read_text())==old
    reader=cache.ShardedReader(directory,root,(32,32));np.testing.assert_array_equal(reader.read(path,4),expected);reader.close()


def test_truncated_shard_rejected(source):
    root,path,directory=source
    m=cache.build_group([path],directory,root,(32,32))[0];stem=cache.stem_for(directory,root,path)
    with (stem.parent/m['shards'][0]).open('r+b') as f:f.truncate(1)
    assert not cache.entry_valid(stem,path,root,(32,32),None,95,0)
    reader=cache.ShardedReader(directory,root,(32,32))
    with pytest.raises(ValueError,match='truncated'):reader.read(path,0)
    reader.close()


def test_chunk_catalog_reuses_maps_and_bounds_descriptors(source,monkeypatch):
    import shutil
    root,path,directory=source
    paths=[]
    for group in range(3):
        group_paths=[]
        for i in range(4):
            other=path.parent.parent/f'chunk-{group+1:03d}'/f'file-{i:03d}.mkv'
            other.parent.mkdir(exist_ok=True);shutil.copyfile(path,other)
            group_paths.append(other)
        cache.build_group(group_paths,directory,root,(32,32))
        paths.extend(group_paths)
    load=np.load;calls=[]
    def counting(*args,**kwargs):calls.append(args[0]);return load(*args,**kwargs)
    monkeypatch.setattr(np,'load',counting)
    reader=cache.ShardedReader(directory,root,(32,32),max_entries=2,max_shards=2)
    for path in paths:
        reader.read(path,0)
        assert len(reader.entries)<=2 and len(reader.catalogs)<=2 and len(reader.shards)<=2
    assert len(calls)==12  # four npy maps per chunk, no per-video npy opens
    reader.close()


def test_optional_serving_roundtrip_matches_uint8_pixels(source):
    import torch
    root,path,directory=source
    pixels=np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)
    tensor=torch.from_numpy(pixels).permute(2,0,1).float()/255
    actual=cache.jpeg_roundtrip_chw(tensor[None],95,0)[0].permute(1,2,0).mul(255).round().to(torch.uint8).numpy()
    np.testing.assert_array_equal(actual,cache.jpeg_roundtrip(pixels,95,0))


def test_parallel_incremental_build_publishes_each_entry_once(source):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    root,path,directory=source
    with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(cache.build_group,[path],directory,root,(32,32)) for _ in range(2)]
        assert sum(len(job.result()) for job in jobs)==1
    reader=cache.ShardedReader(directory,root,(32,32))
    assert reader.read(path,9).shape==(32,32,3)
    reader.close()


def test_cache_symlink_cannot_write_inside_dataset(source,tmp_path):
    root,path,directory=source
    link=tmp_path/'linked-cache';link.symlink_to(root/'would-write-dataset')
    with pytest.raises(ValueError,match='outside the dataset'):
        cache.build_group([path],link,root,(32,32))
    assert not (root/'would-write-dataset').exists()
