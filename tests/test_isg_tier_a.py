"""Tier A probe (scripts/b1k/isg_tier_a.py) on a separate held-out LeRobot root with its own task and episode indices."""

import json
from pathlib import Path
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from b1k_training import parser, train
from test_b1k import tiny_root, write_split  # noqa: F401  (fixture)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/b1k'))
import isg_tier_a  # noqa: E402


def renumbered_copy(root, destination, tasks, episodes):
    """Copy of a LeRobot root whose task and episode indices are renumbered (`tasks`: name -> index,
    `episodes`: old -> new episode_index); frames and videos are unchanged."""
    shutil.copytree(root, destination)
    names = {row['task_index']: row['task'] for row in pq.read_table(root / 'meta/tasks.parquet').to_pylist()}
    pq.write_table(pa.Table.from_pylist([{'task_index': index, 'task': name} for name, index in tasks.items()]),
                   destination / 'meta/tasks.parquet')
    for path in sorted(destination.glob('meta/episodes/*/*.parquet')) + sorted(destination.glob('data/*/*.parquet')):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row['task_index'] = tasks[names[row['task_index']]]
            row['episode_index'] = episodes[row['episode_index']]
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
    return destination


def test_separate_heldout_root_matches_tasks_by_name_and_reproduces_the_same_frames(tiny_root, tmp_path):
    torch.set_num_threads(1)
    split = write_split(tmp_path / 'train_split.json', {'first': [42], 'second': [99]})
    checkpoint = train(parser().parse_args([
        '--dataset-path', str(tiny_root), '--episode-split', str(split), '--batch-size', '2', '--num-workers', '0',
        '--torch-threads', '1', '--device', 'cpu', '--chunk-size', '4', '--image-size', '32', '32', '--hidden-dim',
        '32', '--dim-feedforward', '64', '--enc-layers', '1', '--dec-layers', '1', '--nheads', '4',
        '--no-pretrained-backbone', '--lr', '1e-3', '--lr-backbone', '1e-3', '--regime', 'image', '--goal-fusion',
        'late', '--max-steps', '2', '--output-dir', str(tmp_path / 'run')]))
    probe = ['--device', 'cpu', '--frames-per-task', '4', '--batch-size', '3', '--threads', '1']

    same_root = tmp_path / 'same_root_split.json'
    same_root.write_text(json.dumps({'format': 'isg-episode-split/v1', 'name': 'same-root',
                                     'tasks': {'first': {'held_out_eval': [42]}, 'second': {'held_out_eval': [99]}}}))
    expected = isg_tier_a.run(isg_tier_a.parser().parse_args([str(checkpoint), '--split', str(same_root), *probe]))

    # the test-set layout: its own task indices (in another order) and episode indices, and an eval split
    other = renumbered_copy(tiny_root, tmp_path / 'heldout', {'second': 0, 'first': 1}, {42: 3, 99: 5})
    (other / 'isg_meta').mkdir()
    (other / 'isg_meta/eval_split.json').write_text(json.dumps({
        'format': 'isg-episode-split/v1', 'name': 'eval-tiny',
        'tasks': {'first': {'held_out_eval': [3]}, 'second': {'held_out_eval': [5]}}}))
    result = isg_tier_a.run(isg_tier_a.parser().parse_args([str(checkpoint), '--heldout-dataset', str(other), *probe]))

    assert result['probe']['heldout_dataset'] == str(other.resolve())
    assert result['probe']['split'] == str(other / 'isg_meta/eval_split.json')
    assert result['probe']['split_name'] == 'eval-tiny'
    assert set(result['heldout']['tasks']) == {'first', 'second'}
    assert result['heldout'] == expected['heldout']
    assert result['train'] == expected['train']


def test_separate_heldout_root_needs_a_split_file(tiny_root, tmp_path):
    split = write_split(tmp_path / 'train_split.json', {'first': [42], 'second': [99]})
    checkpoint = train(parser().parse_args([
        '--dataset-path', str(tiny_root), '--episode-split', str(split), '--batch-size', '2', '--num-workers', '0',
        '--torch-threads', '1', '--device', 'cpu', '--chunk-size', '4', '--image-size', '32', '32', '--hidden-dim',
        '32', '--dim-feedforward', '64', '--enc-layers', '1', '--dec-layers', '1', '--nheads', '4',
        '--no-pretrained-backbone', '--regime', 'image', '--goal-fusion', 'late', '--max-steps', '1',
        '--output-dir', str(tmp_path / 'run')]))
    other = renumbered_copy(tiny_root, tmp_path / 'heldout', {'first': 0, 'second': 1}, {42: 0, 99: 1})
    with pytest.raises(SystemExit, match='eval_split.json'):
        isg_tier_a.run(isg_tier_a.parser().parse_args([str(checkpoint), '--heldout-dataset', str(other),
                                                       '--device', 'cpu']))
