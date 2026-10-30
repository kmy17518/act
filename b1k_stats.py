"""Precompute portable ACT normalization JSON from a low-dimensional-only copy."""
import argparse
import json
from pathlib import Path
from b1k_dataset import B1KDataset, expand_task_groups, resolve_episode_split
from b1k_training import resolve_val_split
from b1k_cache_v2 import atomic_json


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task-names',nargs='+')
    p.add_argument('--episode-split',default='auto')
    p.add_argument('--settle-steps',default='0')
    p.add_argument('--val-fraction',type=float,default=.1)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--horizon',type=int,default=100)
    p.add_argument('--normalization',choices=['quantile','zscore'],default='quantile')
    p.add_argument('--action-representation',choices=['relative','absolute'],default='relative')
    p.add_argument('--gripper-action',choices=['absolute','delta'],default='absolute')
    p.add_argument('--estimator',choices=['exact','sampled','auto'],default='exact')
    p.add_argument('--sample-size',type=int,default=10_000_000)
    args=p.parse_args(argv)
    root=args.dataset_root.resolve();output=args.output.resolve()
    if root == output or root in output.parents:
        p.error('statistics output must be outside the source dataset')
    if output.exists():
        raise FileExistsError(output)
    tasks=expand_task_groups(root,args.task_names)
    episode_split=resolve_episode_split(root,args.episode_split,tasks)
    split=resolve_val_split(root,tasks,episode_split,args.val_fraction,args.seed,lowdim_only=True)
    episodes=split['train_episodes'] if split else (episode_split['episodes'] if episode_split else None)
    dataset=B1KDataset(root,tasks,chunk_size=args.horizon,episodes=episodes,settle_steps=args.settle_steps,
                       gripper_state='sum',task_onehot=False,lowdim_only=True)
    try:
        stats=dataset.compute_normalization_stats(args.normalization,action_representation=args.action_representation,
            estimator=args.estimator,gripper_action=args.gripper_action,sample_size=args.sample_size)
        output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,stats)
        print(json.dumps({'frames':stats['count'],'valid_steps':stats['action_count'],'fingerprint':stats['fingerprint']}))
        print(f'RESULT: ACT STATS READY {output}')
    finally:dataset.close()


if __name__=='__main__':main()
