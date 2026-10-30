import json
import os
from pathlib import Path
import numpy as np
import pytest
try:
    from b1k_data_core import StreamingStats, PiFrameBatchSampler, shuffled_positions, fingerprint, source_matches
except ImportError:
    from diffusion_policy.b1k.b1k_data_core import StreamingStats, PiFrameBatchSampler, shuffled_positions, fingerprint, source_matches


def test_exact_quantiles_and_population_moments(tmp_path):
    rng=np.random.default_rng(4); values=rng.normal(size=(1003,5)); values[:,4]=3.
    with StreamingStats(5,len(values),estimator='exact',scratch=tmp_path) as stats:
        for batch in np.array_split(values,13): stats.update(batch)
        result=stats.finish()
        np.testing.assert_allclose(result['mean'],values.mean(0),atol=1e-14)
        np.testing.assert_allclose(result['std'],values.std(0),atol=1e-14)
        np.testing.assert_array_equal(result['q01'],np.quantile(values,.01,axis=0))
        np.testing.assert_array_equal(result['q99'],np.quantile(values,.99,axis=0))
        assert result['std'][4]==0 and result['q01'][4]==result['q99'][4]
    assert not list(tmp_path.iterdir())


def test_sample_is_independent_of_batch_boundaries():
    x=np.random.default_rng(0).normal(size=(10001,3)); outputs=[]
    for pieces in (1,7,37):
        with StreamingStats(3,len(x),estimator='sampled',sample_size=1000) as stats:
            for piece in np.array_split(x,pieces):stats.update(piece)
            outputs.append((stats.finish(),stats.metadata()))
    for output,meta in outputs[1:]:
        for key in ('q01','q99'):np.testing.assert_array_equal(output[key],outputs[0][0][key])
        np.testing.assert_allclose(output['std'],x.std(0),atol=1e-14)
        assert meta['quantile_count']==outputs[0][1]['quantile_count']


@pytest.mark.parametrize('length',[1,2,3,10,31,256,1003])
def test_permutation_has_no_omitted_or_duplicate_frames(length):
    result=shuffled_positions(np.arange(length),length,4,2)
    np.testing.assert_array_equal(np.sort(result),np.arange(length))
    np.testing.assert_array_equal(np.r_[shuffled_positions(np.arange(length//2),length,4,2),shuffled_positions(np.arange(length//2,length),length,4,2)],result)


def test_ddp_global_batches_resume_and_drop_last():
    full=list(PiFrameBatchSampler(103,12,0,20,7))
    ranks=[list(PiFrameBatchSampler(103,12,0,20,7,rank=r,world_size=3)) for r in range(3)]
    for step,batch in enumerate(full):assert batch==sum([r[step] for r in ranks],[])
    assert len(set(sum(full[:8],[])))==96
    assert list(PiFrameBatchSampler(103,12,13,20,7))==full[13:]
    assert len(list(PiFrameBatchSampler(70_486_600,1024,100000,100002,7)))==2


def test_portable_key_ignores_mtime_and_root_but_checks_revision_size(tmp_path):
    roots=[tmp_path/'one',tmp_path/'two']; keys=[]
    for i,root in enumerate(roots):
        (root/'data').mkdir(parents=True); p=root/'data/example.parquet';p.write_bytes(b'abc')
        os.utime(p,ns=(i+1,i+1))
        keys.append(fingerprint(root,[{'episode_index':1}],[p],dataset_revision='abc'))
    assert keys[0]==keys[1]
    link=tmp_path/'linked-root';link.symlink_to(roots[0],target_is_directory=True)
    assert keys[0]==fingerprint(link,[{'episode_index':1}],[link/'data/example.parquet'],dataset_revision='abc')
    p=roots[1]/'data/example.parquet'
    assert source_matches({'source_size':3,'dataset_revision':'abc'},p,roots[1],'abc')
    assert not source_matches({'source_size':3,'dataset_revision':'abc'},p,roots[1],'def')
    p.write_bytes(b'abcd')
    assert keys[0]!=fingerprint(roots[1],[{'episode_index':1}],[p],dataset_revision='abc')


def test_exact_single_valid_step_has_zero_std_and_exact_quantiles():
    x=np.array([[2.,-1.,0.]])
    with StreamingStats(3,1,estimator='exact') as stats:
        stats.update(x);result=stats.finish()
    np.testing.assert_array_equal(result['mean'],x[0])
    np.testing.assert_array_equal(result['std'],np.zeros(3))
    np.testing.assert_array_equal(result['q01'],x[0])
    np.testing.assert_array_equal(result['q99'],x[0])
