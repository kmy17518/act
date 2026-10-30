import pickle
import numpy as np
import pytest
import torch
from test_b1k import tiny_root
from b1k_dataset import B1KDataset, action_reference, extract_state, encode_actions, decode_actions
from b1k_cache_v2 import build_group
from b1k_frame_cache import selected_videos, verify_frame_cache


def test_exact_padding_exclusion_absolute_grippers(tiny_root):
    d=B1KDataset(tiny_root,chunk_size=3,image_size=(16,16),gripper_state='sum',task_onehot=False)
    try:
        stats=d.compute_normalization_stats('quantile',action_representation='relative',estimator='exact',gripper_action='absolute')
        table=d._ensure_table();all_values=[]
        for pos,ep in enumerate(d.episodes):
            start,end=int(d.starts[pos]),int(d.ends[pos])
            state=extract_state(table['state'][start:end],'sum')
            reference=action_reference(state,stats['action_delta'])
            for anchor in range(end-start):
                all_values.extend(table['action'][start+anchor:min(start+anchor+3,end)].astype(np.float64)-reference[anchor])
        values=np.asarray(all_values)
        assert stats['action_count']==24
        np.testing.assert_array_equal(stats['action_q01'],np.quantile(values,.01,axis=0))
        np.testing.assert_allclose(stats['action_std'],values.std(0),atol=1e-12)
        assert stats['action_delta']['gripper_action_dims']==[] and stats['std_eps']==1e-6
        state=extract_state(table['state'][:2],'sum')
        raw=table['action'][:2,None,:]
        encoded=encode_actions(raw,state,stats)
        np.testing.assert_allclose(decode_actions(encoded,stats,state),raw,atol=1e-5)
    finally:d.close()


def test_sidecar_lazy_goals_and_spawn_equivalence(tiny_root,tmp_path):
    native=B1KDataset(tiny_root,chunk_size=3,image_size=(16,16),goal_views=['zed_link'],gripper_state='sum')
    directory=tmp_path/'cache'
    for path in selected_videos(native):build_group([path],directory,tiny_root,(16,16))
    assert verify_frame_cache(native,directory,samples=4,log=lambda *_:None)==12
    cached=B1KDataset(tiny_root,chunk_size=3,image_size=(16,16),frame_cache=directory,goal_views=['zed_link'],gripper_state='sum')
    cached.stats = native.compute_normalization_stats(estimator='exact',gripper_action='absolute')
    try:
        assert isinstance(cached._table['state'],np.memmap)
        assert not isinstance(cached.goal_table,np.ndarray)
        assert len(pickle.dumps(cached))<100_000
        restored=pickle.loads(pickle.dumps(cached))
        for i in range(len(cached)):
            first,second=cached[i],restored[i]
            for a,b in zip(first,second):torch.testing.assert_close(a,b,rtol=0,atol=0)
        restored.close()
        # The cache remains usable by a spawned loader and does not pickle lowdim/image arrays.
        loader=torch.utils.data.DataLoader(cached,batch_size=3,num_workers=1,multiprocessing_context='spawn')
        assert sum(len(batch[0]) for batch in loader)==len(cached)
    finally:native.close();cached.close()


def test_v2_training_resume_absolute_grippers(tiny_root,tmp_path):
    from b1k_training import train,parser,load_checkpoint
    native=B1KDataset(tiny_root,chunk_size=4,image_size=(32,32))
    cache=tmp_path/'cache'
    for path in selected_videos(native):build_group([path],cache,tiny_root,(32,32))
    import json
    stats_path=tmp_path/'precomputed.json'
    stats_path.write_text(json.dumps(native.compute_normalization_stats('quantile',action_representation='relative',estimator='exact',gripper_action='absolute')))
    native.close()
    common=['--dataset-path',str(tiny_root),'--batch-size','2','--num-workers','0','--device','cpu',
            '--chunk-size','4','--image-size','32','32','--hidden-dim','32','--dim-feedforward','64',
            '--enc-layers','1','--dec-layers','1','--nheads','4','--no-pretrained-backbone','--save-every','1',
            '--frame-cache',str(cache),'--stats-file',str(stats_path),'--goal-views','zed_link','--goal-fusion','late',
            '--action-representation','relative','--normalization','quantile']
    full=tmp_path/'full';split=tmp_path/'split'
    train(parser().parse_args(common+['--output-dir',str(full),'--max-steps','3']))
    train(parser().parse_args(common+['--output-dir',str(split),'--max-steps','2']))
    path=train(parser().parse_args(common+['--output-dir',str(split),'--max-steps','3','--resume',str(split/'step_00000002.pt')]))
    expected=load_checkpoint(full/'step_00000003.pt');actual=load_checkpoint(path)
    for key in expected['model']:torch.testing.assert_close(expected['model'][key],actual['model'][key],rtol=0,atol=0)
    assert actual['normalization']['gripper_action']=='absolute'
    assert actual['normalization']['estimator']=='exact'


def test_precompute_from_lowdim_copy_matches_rgb_selection(tiny_root,tmp_path):
    import json,shutil
    from b1k_stats import main
    copy=tmp_path/'lowdim';shutil.copytree(tiny_root,copy,ignore=shutil.ignore_patterns('videos'))
    path=tmp_path/'stats.json'
    main(['--dataset-root',str(copy),'--output',str(path),'--episode-split','none','--settle-steps','all',
          '--val-fraction','0','--horizon','4'])
    rgb=B1KDataset(tiny_root,chunk_size=4,gripper_state='sum',task_onehot=False)
    try:
        stats=json.loads(path.read_text())
        assert stats['fingerprint']==rgb.fingerprint() and stats['count']==len(rgb)
        expected=rgb.compute_normalization_stats('quantile',action_representation='relative',estimator='exact',gripper_action='absolute')
        np.testing.assert_array_equal(stats['action_q01'],expected['action_q01'])
    finally:rgb.close()
