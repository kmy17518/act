#!/usr/bin/env python3
"""Merge several LeRobot v3.0 roots that share one robot into one multi-task root, without re-encoding video.

Built for the goal-image conditioning experiments: the two skill-segment datasets
`2026-challenge-demos-radio-nav-goal` and `2026-challenge-demos-radio-pickup-goal` (one task each, both cut
from the `turning_on_radio` demos and both shipping byte-identical copies of that task's packed camera videos)
become one root with two tasks, so every trainer in this workspace (ACT, Diffusion Policy, openpi) can train on
the mixture through its ordinary single-root loader, including the unmodified `my` and language branches.

Layout of the output (input root i, in command-line order):

    data/chunk-<i>/file-XXX.parquet        rows of root i with `episode_index`, `index` and `task_index`
                                           rewritten to the merged numbering; one row group per episode
    meta/episodes/chunk-000/file-000.parquet   all episodes; `data/chunk_index` = i; video chunk indices below
    meta/tasks.parquet, meta/tasks.jsonl   merged task table (task names must be unique per task; a task name
                                           present in several roots is one merged task)
    meta/info.json                         root 0's info with merged totals (features must be identical)
    meta/stats.json                        the per-root `stats.json` aggregated with lerobot's parallel formula
                                           (exact min/max/mean/std/count; quantiles count-weighted, as lerobot)
    videos/<key>/chunk-<c>/file-XXX.mp4    hard links (or symlinks/copies, --link): keys named by
                                           --shared-video-prefix are assumed identical across roots and are
                                           linked ONCE from --shared-video-root (chunk index kept, so a frame
                                           cache built for that root stays valid: same inode, size and mtime);
                                           every other key gets chunk-<i> per root
    goal_images/<key>/episode-XXXXXX.png   renumbered links when the roots ship them
    MERGE_MANIFEST.json                    provenance (inputs, offsets, task map, link mode, checks)

Numeric rows are copied verbatim apart from the three index columns; `frame_index`/`timestamp` restart per
episode as before. `--verify` re-reads the output and compares sampled rows/episodes against the inputs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FORMAT = 'b1k_merged_lerobot_root_v1'
INDEX_COLUMNS = ('episode_index', 'index', 'task_index')


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def episode_tables(root, info):
    files = sorted((root / 'meta/episodes').glob('*/*.parquet'))
    if not files:
        raise FileNotFoundError(f'{root}: no meta/episodes/*/*.parquet')
    return pa.concat_tables([pq.read_table(path) for path in files], promote_options='default')


def task_rows(root):
    table = pq.read_table(root / 'meta/tasks.parquet')
    name_column = next((c for c in ('task', '__index_level_0__', 'task_name') if c in table.schema.names), None)
    if name_column is None or 'task_index' not in table.schema.names:
        raise ValueError(f'{root}: meta/tasks.parquet needs task_index and task columns')
    rows = {int(i): str(name) for i, name in zip(table['task_index'].to_pylist(), table[name_column].to_pylist())}
    if len(set(rows.values())) != len(rows):
        raise ValueError(f'{root}: duplicate task names')
    descriptions = {}
    sidecar = root / 'meta/tasks.jsonl'
    if sidecar.is_file():
        for line in sidecar.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                descriptions[str(row['task_name'])] = str(row['task'])
    return rows, descriptions, table.schema.metadata


def link(source, target, mode):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == 'hard':
        os.link(source, target)
    elif mode == 'symlink':
        target.symlink_to(source.resolve())
    elif mode == 'copy':
        shutil.copy2(source, target)
    else:
        raise ValueError(mode)


def sampled_digest(path, chunk=1 << 20):
    """Cheap identity check for large videos: size plus SHA-256 of the first/middle/last MiB."""
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for offset in sorted({0, max(0, size // 2 - chunk // 2), max(0, size - chunk)}):
            stream.seek(offset)
            digest.update(stream.read(chunk))
    return size, digest.hexdigest()


def aggregate_stats(stats_list):
    """lerobot.datasets.compute_stats.aggregate_stats, inlined (same arithmetic) so this tool has no lerobot dependency."""
    keys = set().union(*(s.keys() for s in stats_list))
    result = {}
    for key in sorted(keys):
        entries = [s[key] for s in stats_list if key in s]
        means = np.stack([np.asarray(e['mean'], dtype=np.float64) for e in entries])
        variances = np.stack([np.asarray(e['std'], dtype=np.float64) ** 2 for e in entries])
        counts = np.stack([np.asarray(e['count'], dtype=np.float64) for e in entries])
        total_count = counts.sum(axis=0)
        while counts.ndim < means.ndim:
            counts = np.expand_dims(counts, axis=-1)
        total_mean = (means * counts).sum(axis=0) / total_count
        total_variance = ((variances + (means - total_mean) ** 2) * counts).sum(axis=0) / total_count
        aggregated = {'min': np.min(np.stack([np.asarray(e['min'], dtype=np.float64) for e in entries]), axis=0),
                      'max': np.max(np.stack([np.asarray(e['max'], dtype=np.float64) for e in entries]), axis=0),
                      'mean': total_mean, 'std': np.sqrt(total_variance), 'count': total_count}
        for q_key in [k for k in entries[0] if k.startswith('q') and k[1:].isdigit()]:
            if all(q_key in e for e in entries):
                values = np.stack([np.asarray(e[q_key], dtype=np.float64) for e in entries])
                aggregated[q_key] = (values * counts).sum(axis=0) / total_count
        result[key] = {k: (v.astype(np.int64) if k == 'count' else v).tolist() for k, v in aggregated.items()}
    return result


def merge(args):
    output = Path(args.output).resolve()
    roots = [Path(r).resolve() for r in args.root]
    if len(roots) < 2:
        raise ValueError('Give at least two --root inputs')
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f'{output} exists; pass --overwrite to replace it')
        shutil.rmtree(output)
    infos = [read_json(root / 'meta/info.json') for root in roots]
    for root, info in zip(roots, infos):
        if info.get('codebase_version') != 'v3.0':
            raise ValueError(f'{root}: expected LeRobot v3.0, got {info.get("codebase_version")}')
        if info['features'] != infos[0]['features'] or info['fps'] != infos[0]['fps'] or \
                info.get('robot_type') != infos[0].get('robot_type'):
            raise ValueError(f'{root}: features/fps/robot_type differ from {roots[0]}')
    video_keys = [key for key, feature in infos[0]['features'].items() if feature['dtype'] == 'video']
    shared_prefixes = tuple(args.shared_video_prefix or ())
    shared_root = Path(args.shared_video_root).resolve() if args.shared_video_root else None
    if shared_prefixes and shared_root is None:
        raise ValueError('--shared-video-prefix requires --shared-video-root')
    shared_keys = [key for key in video_keys if key.startswith(shared_prefixes)] if shared_prefixes else []

    # ---- tasks -------------------------------------------------------------------------------------------------
    global_tasks, descriptions, tasks_metadata = {}, {}, None
    per_root_task_maps = []
    for root in roots:
        rows, root_descriptions, metadata = task_rows(root)
        tasks_metadata = tasks_metadata or metadata
        mapping = {}
        for local_index, name in sorted(rows.items()):
            if name not in global_tasks:
                global_tasks[name] = len(global_tasks)
            mapping[local_index] = global_tasks[name]
            if name in root_descriptions:
                if descriptions.get(name, root_descriptions[name]) != root_descriptions[name]:
                    raise ValueError(f'Conflicting descriptions for task {name!r}')
                descriptions[name] = root_descriptions[name]
        per_root_task_maps.append(mapping)
    log(f'tasks: {global_tasks}')

    # ---- episodes + data ---------------------------------------------------------------------------------------
    episode_offset = frame_offset = 0
    episode_tables_out, manifest_roots, video_plan = [], [], {}
    for i, (root, info) in enumerate(zip(roots, infos)):
        episodes = episode_tables(root, info)
        episodes = episodes.sort_by('episode_index')
        local_ids = episodes['episode_index'].to_numpy()
        if not np.array_equal(local_ids, np.arange(len(local_ids))):
            raise ValueError(f'{root}: expected contiguous episode_index 0..N-1 (got {local_ids[:5]}...)')
        lengths = episodes['length'].to_numpy()
        starts = episodes['dataset_from_index'].to_numpy()
        ends = episodes['dataset_to_index'].to_numpy()
        if not (np.array_equal(ends - starts, lengths) and starts[0] == 0 and np.array_equal(starts[1:], ends[:-1])):
            raise ValueError(f'{root}: dataset_from/to_index are not contiguous')
        total_frames = int(ends[-1])
        if total_frames != info['total_frames']:
            raise ValueError(f'{root}: info.json total_frames {info["total_frames"]} != metadata {total_frames}')
        task_column = episodes['task_index'].to_numpy() if 'task_index' in episodes.schema.names else None
        if task_column is None:
            raise ValueError(f'{root}: episodes metadata needs task_index')
        mapping = per_root_task_maps[i]
        new_columns = {
            'episode_index': pa.array(local_ids + episode_offset, type=episodes.schema.field('episode_index').type),
            'dataset_from_index': pa.array(starts + frame_offset, type=episodes.schema.field('dataset_from_index').type),
            'dataset_to_index': pa.array(ends + frame_offset, type=episodes.schema.field('dataset_to_index').type),
            'task_index': pa.array([mapping[int(t)] for t in task_column], type=episodes.schema.field('task_index').type),
            'data/chunk_index': pa.array(np.full(len(local_ids), i), type=episodes.schema.field('data/chunk_index').type),
            'meta/episodes/chunk_index': pa.array(np.zeros(len(local_ids), dtype=np.int64),
                                                  type=episodes.schema.field('meta/episodes/chunk_index').type),
            'meta/episodes/file_index': pa.array(np.zeros(len(local_ids), dtype=np.int64),
                                                 type=episodes.schema.field('meta/episodes/file_index').type),
        }
        # Videos: shared keys keep their chunk index (files linked once from the shared root); others move to chunk-i.
        for key in video_keys:
            chunk_field = f'videos/{key}/chunk_index'
            file_field = f'videos/{key}/file_index'
            chunks = episodes[chunk_field].to_numpy()
            files = episodes[file_field].to_numpy()
            plan = video_plan.setdefault(key, {})
            if key in shared_keys:
                for chunk, file in set(zip(chunks.tolist(), files.tolist())):
                    relative = Path(info['video_path'].format(video_key=key, chunk_index=chunk, file_index=file))
                    own = root / relative
                    shared = shared_root / relative
                    if not shared.is_file():
                        raise FileNotFoundError(f'shared video missing: {shared}')
                    if not own.is_file():
                        raise FileNotFoundError(f'{root}: video missing: {own}')
                    if sampled_digest(own) != sampled_digest(shared):
                        raise ValueError(f'{own} is not identical to the shared {shared}; drop --shared-video-prefix '
                                         f'for {key} or fix the inputs')
                    plan[str(relative)] = shared
            else:
                new_columns[chunk_field] = pa.array(np.full(len(local_ids), i), type=episodes.schema.field(chunk_field).type)
                for chunk, file in set(zip(chunks.tolist(), files.tolist())):
                    source = root / info['video_path'].format(video_key=key, chunk_index=chunk, file_index=file)
                    if not source.is_file():
                        raise FileNotFoundError(f'{root}: video missing: {source}')
                    target = info['video_path'].format(video_key=key, chunk_index=i, file_index=file)
                    plan[target] = source
        for name, column in new_columns.items():
            episodes = episodes.set_column(episodes.schema.get_field_index(name), name, column)
        episode_tables_out.append(episodes)

        # Data files: rewrite index columns, one row group per episode. The input's own data/chunk_index and
        # data/file_index (read again from its metadata, before the rewrite above) locate the source files.
        original = episode_tables(root, info)
        data_files = sorted({(int(c), int(f)) for c, f in zip(original['data/chunk_index'].to_numpy(),
                                                             original['data/file_index'].to_numpy())})
        rows_written = 0
        for chunk, file in data_files:
            source = root / info['data_path'].format(chunk_index=chunk, file_index=file)
            table = pq.read_table(source)
            local_episode = table['episode_index'].to_numpy()
            new_index = table['index'].to_numpy() + frame_offset
            new_task = np.asarray([mapping[int(t)] for t in table['task_index'].to_numpy()])
            table = table.set_column(table.schema.get_field_index('episode_index'), 'episode_index',
                                     pa.array(local_episode + episode_offset, type=table.schema.field('episode_index').type))
            table = table.set_column(table.schema.get_field_index('index'), 'index',
                                     pa.array(new_index, type=table.schema.field('index').type))
            table = table.set_column(table.schema.get_field_index('task_index'), 'task_index',
                                     pa.array(new_task, type=table.schema.field('task_index').type))
            target = output / infos[0]['data_path'].format(chunk_index=i, file_index=file)
            target.parent.mkdir(parents=True, exist_ok=True)
            order = np.argsort(table['index'].to_numpy(), kind='stable')
            table = table.take(pa.array(order))
            episode_column = table['episode_index'].to_numpy()
            boundaries = np.flatnonzero(np.diff(episode_column)) + 1
            with pq.ParquetWriter(target, table.schema, compression='snappy') as writer:
                for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, len(table)]):
                    writer.write_table(table.slice(start, stop - start))
            rows_written += table.num_rows
        if rows_written != total_frames:
            raise ValueError(f'{root}: wrote {rows_written} rows but metadata declares {total_frames} frames')
        manifest_roots.append({'root': str(root), 'episodes': int(len(local_ids)), 'frames': total_frames,
                               'episode_offset': episode_offset, 'frame_offset': frame_offset,
                               'data_chunk': i, 'task_index_map': {str(k): v for k, v in mapping.items()}})
        log(f'root {i}: {root.name}: {len(local_ids)} episodes, {total_frames} frames -> episodes '
            f'{episode_offset}..{episode_offset + len(local_ids) - 1}, data chunk-{i:03d}')
        episode_offset += len(local_ids)
        frame_offset += total_frames

    # ---- links --------------------------------------------------------------------------------------------------
    linked = 0
    for key, plan in video_plan.items():
        for relative, source in plan.items():
            link(source, output / relative, args.link)
            linked += 1
    log(f'linked {linked} video files ({args.link}); shared keys: {shared_keys}')
    goal_dirs = [root / 'goal_images' for root in roots]
    if all(d.is_dir() for d in goal_dirs):
        for entry in manifest_roots:
            root = Path(entry['root'])
            for key_dir in sorted((root / 'goal_images').iterdir()):
                for png in sorted(key_dir.glob('episode-*.png')):
                    local = int(png.stem.split('-')[1])
                    link(png, output / 'goal_images' / key_dir.name / f'episode-{local + entry["episode_offset"]:06d}.png',
                         args.link)
        log('linked goal_images with merged episode numbers')
    for root in roots:
        annotations = root / 'annotations'
        if annotations.is_dir():
            for path in annotations.rglob('*.json'):
                target = output / path.relative_to(root)
                if not target.exists():
                    link(path, target, args.link)
    for name in ('LICENSE', 'meta/modality.json'):
        source = roots[0] / name
        if source.is_file():
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output / name)

    # ---- meta ---------------------------------------------------------------------------------------------------
    episodes_out = pa.concat_tables(episode_tables_out, promote_options='default')
    meta_path = output / 'meta/episodes/chunk-000/file-000.parquet'
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(episodes_out, meta_path)
    names = list(global_tasks)
    tasks_table = pa.table({'task_index': pa.array([global_tasks[n] for n in names], type=pa.int64()),
                            'task': pa.array(names, type=pa.string())})
    if tasks_metadata:
        tasks_table = tasks_table.replace_schema_metadata(tasks_metadata)
    pq.write_table(tasks_table, output / 'meta/tasks.parquet')
    with (output / 'meta/tasks.jsonl').open('w') as stream:
        for name in names:
            stream.write(json.dumps({'task_index': global_tasks[name], 'task_name': name,
                                     'task': descriptions.get(name, name)}) + '\n')
    info = dict(infos[0])
    info.update(total_episodes=episode_offset, total_frames=frame_offset, total_tasks=len(global_tasks),
                splits={'train': f'0:{episode_offset}'})
    (output / 'meta/info.json').write_text(json.dumps(info, indent=4))
    stats = [read_json(root / 'meta/stats.json') for root in roots if (root / 'meta/stats.json').is_file()]
    if len(stats) == len(roots):
        (output / 'meta/stats.json').write_text(json.dumps(aggregate_stats(stats), indent=4))
    manifest = {'format': FORMAT, 'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'command': sys.argv, 'link': args.link, 'roots': manifest_roots, 'tasks': global_tasks,
                'shared_video_root': str(shared_root) if shared_root else None, 'shared_video_keys': shared_keys,
                'total_episodes': episode_offset, 'total_frames': frame_offset,
                'stats_json': 'lerobot aggregate_stats formula over the input stats.json files (quantiles count-weighted)'
                if len(stats) == len(roots) else 'not written (an input lacks meta/stats.json)'}
    (output / 'MERGE_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    (output / 'README.md').write_text(
        f'# Merged LeRobot v3.0 root\n\nBuilt by `merge_lerobot_roots.py` from {len(roots)} roots '
        f'({", ".join(r.name for r in roots)}): {episode_offset} episodes, {frame_offset} frames, '
        f'{len(global_tasks)} tasks {names}. Video files are {args.link} links to the inputs; see MERGE_MANIFEST.json.\n')
    log(f'wrote {output}: {episode_offset} episodes, {frame_offset} frames, {len(global_tasks)} tasks')
    return output


def verify(output, sample_episodes=8, seed=0):
    """Independent re-read of the merged root against the inputs recorded in MERGE_MANIFEST.json."""
    output = Path(output)
    manifest = read_json(output / 'MERGE_MANIFEST.json')
    info = read_json(output / 'meta/info.json')
    episodes = pq.read_table(output / 'meta/episodes/chunk-000/file-000.parquet').sort_by('episode_index')
    ids = episodes['episode_index'].to_numpy()
    assert np.array_equal(ids, np.arange(manifest['total_episodes'])), 'episode_index not contiguous'
    assert int(episodes['dataset_to_index'].to_numpy()[-1]) == manifest['total_frames'] == info['total_frames']
    tasks = pq.read_table(output / 'meta/tasks.parquet').to_pylist()
    assert {row['task']: row['task_index'] for row in tasks} == manifest['tasks'], 'tasks.parquet mismatch'
    rng = np.random.default_rng(seed)
    checks = 0
    for entry in manifest['roots']:
        root = Path(entry['root'])
        source_episodes = episode_tables(root, read_json(root / 'meta/info.json')).sort_by('episode_index')
        chosen = rng.choice(entry['episodes'], size=min(sample_episodes, entry['episodes']), replace=False)
        for local in sorted(int(c) for c in chosen):
            merged_row = episodes.slice(local + entry['episode_offset'], 1).to_pylist()[0]
            source_row = source_episodes.slice(local, 1).to_pylist()[0]
            assert merged_row['length'] == source_row['length']
            assert merged_row['dataset_from_index'] == source_row['dataset_from_index'] + entry['frame_offset']
            assert merged_row['task_index'] == entry['task_index_map'][str(source_row['task_index'])]
            assert merged_row['tasks'] == source_row['tasks']
            merged_data = pq.read_table(output / info['data_path'].format(chunk_index=merged_row['data/chunk_index'],
                                                                           file_index=merged_row['data/file_index']))
            merged_data = merged_data.filter(pc.equal(merged_data['episode_index'], merged_row['episode_index']))
            source_data = pq.read_table(root / info['data_path'].format(chunk_index=source_row['data/chunk_index'],
                                                                         file_index=source_row['data/file_index']))
            source_data = source_data.filter(pc.equal(source_data['episode_index'], local))
            assert merged_data.num_rows == source_data.num_rows == merged_row['length']
            for column in ('observation.state', 'action', 'timestamp', 'frame_index'):
                assert merged_data[column].to_pylist() == source_data[column].to_pylist(), column
            assert merged_data['index'].to_pylist() == [v + entry['frame_offset'] for v in source_data['index'].to_pylist()]
            assert set(merged_data['task_index'].to_pylist()) == {merged_row['task_index']}
            for key in [k for k, f in info['features'].items() if f['dtype'] == 'video']:
                merged_video = output / info['video_path'].format(
                    video_key=key, chunk_index=merged_row[f'videos/{key}/chunk_index'],
                    file_index=merged_row[f'videos/{key}/file_index'])
                source_video = root / info['video_path'].format(
                    video_key=key, chunk_index=source_row[f'videos/{key}/chunk_index'],
                    file_index=source_row[f'videos/{key}/file_index'])
                assert merged_video.is_file(), merged_video
                assert sampled_digest(merged_video) == sampled_digest(source_video), (merged_video, source_video)
                for field in ('from_timestamp', 'to_timestamp'):
                    assert merged_row[f'videos/{key}/{field}'] == source_row[f'videos/{key}/{field}']
            checks += 1
    log(f'verify: {checks} sampled episodes match their inputs (rows, indices, tasks, video files, timestamps)')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', required=True)
    parser.add_argument('--root', action='append', help='Input LeRobot v3.0 root (repeat, order defines chunks)')
    parser.add_argument('--shared-video-root', help='Root whose video files the inputs duplicate for the shared keys')
    parser.add_argument('--shared-video-prefix', action='append',
                        help='Video key prefix (e.g. observation.rgb.) whose files are linked once from --shared-video-root')
    parser.add_argument('--link', choices=['hard', 'symlink', 'copy'], default='hard')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verify', action='store_true', help='Re-read the output and compare samples with the inputs')
    parser.add_argument('--verify-only', action='store_true', help='Only verify an existing --output')
    args = parser.parse_args()
    if args.verify_only:
        verify(args.output)
        return
    output = merge(args)
    if args.verify:
        verify(output)


if __name__ == '__main__':
    main()
