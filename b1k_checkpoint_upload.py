"""CPU-only, single-writer checkpoint publication for dedicated Hugging Face repos."""

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
import traceback
import uuid


MARKER = '.b1k-uploader.json'
STEP = re.compile(r'^step[-_](\d{8})\.pt$')
PROTOCOL = 1


class SafetyError(RuntimeError):
    """An ownership or integrity failure that must not be retried automatically."""


class RetryableError(RuntimeError):
    """An operation whose durable intent can be retried."""


def redact_credentials(value):
    text = str(value)
    for name in ('HF_TOKEN', 'WANDB_API_KEY'):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, '[REDACTED]')
    # Retry errors can contain relative signed URLs as well as absolute URLs.
    text = re.sub(r'''\?[^\s'"<>)]*''', '?[REDACTED]', text)
    text = re.sub(r'(?i)\b(Bearer|Basic)\s+[^\s\'"<>),;]+', r'\1 [REDACTED]', text)
    return re.sub(r'\bhf_[A-Za-z0-9]+\b', '[REDACTED]', text)


class CredentialFilter(logging.Filter):
    def filter(self, record):
        record.msg = redact_credentials(record.getMessage())
        record.args = ()
        if record.exc_info:
            record.exc_text = redact_credentials(''.join(traceback.format_exception(*record.exc_info)))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = redact_credentials(record.exc_text)
        if record.stack_info:
            record.stack_info = redact_credentials(record.stack_info)
        return True


def configure_safe_logging():
    redactor = CredentialFilter()
    loggers = [logging.getLogger()]
    loggers.extend(logger for name, logger in logging.Logger.manager.loggerDict.items()
                   if isinstance(logger, logging.Logger)
                   and name.split('.')[0] in ('huggingface_hub', 'urllib3', 'requests'))
    for logger in loggers:
        logger.addFilter(redactor)
        for handler in logger.handlers:
            handler.addFilter(redactor)
    if logging.lastResort:
        logging.lastResort.addFilter(redactor)


def now():
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_dir(path.parent)


def git_oid(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def digest(path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    git = hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
            git.update(block)
    return {'size': size, 'sha256': sha.hexdigest(), 'git_oid': git.hexdigest()}


@contextmanager
def exclusive_lock(path):
    with Path(path).open('a+b') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SafetyError(f'Another uploader holds {path}') from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def make_api(http_timeout=120):
    # The requests backend also bounds HF endpoints that omit a timeout themselves.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    import requests
    from huggingface_hub import HfApi, configure_http_backend, constants

    constants.HF_HUB_DISABLE_XET = True
    constants.HF_HUB_ENABLE_HF_TRANSFER = False

    class BoundedSession(requests.Session):
        def request(self, method, url, **kwargs):
            if kwargs.get('timeout') is None:
                kwargs['timeout'] = (10, http_timeout)
            return super().request(method, url, **kwargs)

    configure_http_backend(backend_factory=BoundedSession)
    configure_safe_logging()
    return HfApi(token=os.environ.get('HF_TOKEN'))


class CheckpointUploader:
    """Keep all remote mutations recoverable from an fsynced local journal."""

    def __init__(self, run_dir, staging_dir, repo_id, api, *, run_id=None,
                 max_steps=300000, eval_every=10000, metadata=None):
        self.run_dir = Path(run_dir).resolve()
        self.staging_dir = Path(staging_dir).resolve()
        self.repo_id = repo_id
        self.api = api
        self.run_id = run_id or self.run_dir.name
        self.max_steps = max_steps
        self.eval_every = eval_every
        self.metadata = metadata or {}
        if max_steps <= 0 or eval_every <= 0 or max_steps % eval_every:
            raise ValueError('max_steps must be a positive multiple of eval_every')
        self.state_path = self.staging_dir / 'state.json'
        self.status_path = self.staging_dir / 'status.json'
        self.state = None
        self._locks = None
        self._authenticated = False

    def __enter__(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        if self.run_dir.stat().st_dev != self.staging_dir.stat().st_dev:
            raise SafetyError('run-dir and staging-dir must share a filesystem for durable hardlinks')
        self._locks = ExitStack()
        try:
            self._locks.enter_context(exclusive_lock(self.run_dir / '.checkpoint-upload.lock'))
            self._locks.enter_context(exclusive_lock(self.staging_dir / 'writer.lock'))
            (self.staging_dir / 'files').mkdir(exist_ok=True)
            config = {'repo_id': self.repo_id, 'run_id': self.run_id,
                      'run_dir': str(self.run_dir), 'max_steps': self.max_steps,
                      'eval_every': self.eval_every, 'metadata': self.metadata}
            if self.state_path.exists():
                self.state = json.loads(self.state_path.read_text())
                if self.state.get('config') != config or self.state.get('protocol') != PROTOCOL:
                    raise SafetyError('Staging journal belongs to a different run or configuration')
            else:
                self.state = {'protocol': PROTOCOL, 'config': config,
                              'writer_uuid': str(uuid.uuid4()), 'phase': 'new',
                              'head': None, 'manifest': None, 'pending': None,
                              'gc_freed_bytes': 0, 'gc_freed_oids': [], 'transactions': [],
                              'last_success': None, 'error_count': 0, 'last_error': None}
                self.save()
            self.status('starting')
            return self
        except BaseException:
            self._locks.close()
            raise

    def __exit__(self, *args):
        return self._locks.__exit__(*args)

    def save(self):
        save_json(self.state_path, self.state)

    def status(self, health, error=None):
        manifest = self.state.get('manifest') or {}
        full = manifest.get('full') or {}
        value = {'updated_at': now(), 'pid': os.getpid(), 'health': health,
                 'repo_id': self.repo_id, 'run_id': self.run_id,
                 'writer_uuid': self.state['writer_uuid'],
                 'current_step': full.get('step', 0),
                 'eval_steps': sorted(int(step) for step in manifest.get('eval', {})),
                 'last_success': self.state['last_success'],
                 'last_error': error or self.state['last_error'],
                 'error_count': self.state['error_count'],
                 'gc_freed_bytes': self.state['gc_freed_bytes'],
                 'gc_freed_oids': self.state['gc_freed_oids'],
                 'pending_phase': (self.state.get('pending') or {}).get('phase'),
                 'head': self.state['head']}
        save_json(self.status_path, value)

    def record_error(self, exc):
        self.state['error_count'] += 1
        self.state['last_error'] = redact_credentials(f'{type(exc).__name__}: {exc}')
        self.save()
        self.status('fatal' if isinstance(exc, SafetyError) else 'retrying')

    def authenticate(self):
        if not self._authenticated:
            from huggingface_hub.errors import HfHubHTTPError
            owner = self.repo_id.split('/')[0]
            try:
                identity = self.api.whoami()
            except HfHubHTTPError as exc:
                if exc.response is not None and exc.response.status_code in (401, 403):
                    raise SafetyError('HF authentication/ownership verification denied') from exc
                raise
            if identity['name'] != owner:
                raise SafetyError(f'Authenticated HF user must own dedicated repo {self.repo_id}')
            self._authenticated = True

    def refs(self):
        refs = self.api.list_repo_refs(self.repo_id, repo_type='model', include_pull_requests=True)
        branches = list(refs.branches)
        if (len(branches) != 1 or branches[0].ref != 'refs/heads/main'
                or refs.tags or getattr(refs, 'converts', [])
                or getattr(refs, 'pull_requests', [])):
            raise SafetyError('Dedicated repo must have only main: unexpected branches, tags, PR or conversion refs')
        return branches[0].target_commit

    def tree(self, head):
        result = {}
        for item in self.api.list_repo_tree(self.repo_id, repo_type='model',
                                           revision=head, recursive=True):
            if not hasattr(item, 'blob_id'):
                continue
            result[item.path] = {'size': item.size, 'git_oid': item.blob_id,
                                 'sha256': item.lfs.sha256 if item.lfs else None}
        return result

    def manifest(self, transaction):
        return {'protocol': PROTOCOL, 'writer_uuid': self.state['writer_uuid'],
                'run_id': self.run_id, 'repo_id': self.repo_id,
                'max_steps': self.max_steps, 'eval_every': self.eval_every,
                'metadata': self.metadata, 'transaction': transaction,
                'eval': {}, 'full': None}

    def assert_tree(self, tree, manifest):
        expected = {MARKER, *self.state['base_tree']}
        managed = list(manifest['eval'].values())
        if manifest['full']:
            managed.append(manifest['full'])
        expected.update(entry['path'] for entry in managed)
        if set(tree) != expected:
            raise SafetyError(f'Unexpected live repository paths: {sorted(set(tree) ^ expected)}')
        marker = json_bytes(manifest)
        if tree[MARKER] != {'size': len(marker), 'git_oid': git_oid(marker), 'sha256': None}:
            raise SafetyError('Remote ownership/transaction marker mismatch')
        for path, info in self.state['base_tree'].items():
            if tree[path] != info:
                raise SafetyError(f'Foreign modification of {path}')
        for entry in managed:
            actual = tree[entry['path']]
            correct_hash = (actual['sha256'] == entry['sha256'] if actual['sha256']
                            else actual['git_oid'] == entry['git_oid'])
            if actual['size'] != entry['size'] or not correct_hash:
                raise SafetyError(f'Remote checkpoint hash mismatch: {entry["path"]}')

    def verified_head(self, manifest, expected_head, allow_rewrite=False):
        head = self.refs()
        tree = self.tree(head)
        self.assert_tree(tree, manifest)
        if head != expected_head:
            if not allow_rewrite:
                raise SafetyError(f'Unexpected writer/head: expected {expected_head}, got {head}')
            latest = next(iter(self.api.list_repo_commits(self.repo_id, repo_type='model', revision='main')))
            if latest.commit_id != head or latest.title != self.commit_title(manifest):
                raise SafetyError('GC recovery found an unexpected writer/commit')
        if self.refs() != head:
            raise SafetyError('Repository head changed during verification')
        return head, tree

    @staticmethod
    def commit_title(manifest):
        return 'b1k uploader ' + manifest['transaction']

    def commit(self, operations, manifest, parent):
        from huggingface_hub import CommitOperationAdd
        operations.append(CommitOperationAdd(path_in_repo=MARKER, path_or_fileobj=json_bytes(manifest)))
        return self.api.create_commit(self.repo_id, repo_type='model', revision='main',
                                      parent_commit=parent, operations=operations,
                                      commit_message=self.commit_title(manifest), num_threads=1).oid

    def initialize(self):
        from huggingface_hub.errors import RepositoryNotFoundError
        if self.state['phase'] == 'ready':
            return
        if self.state['phase'] == 'new':
            try:
                self.api.repo_info(self.repo_id, repo_type='model')
            except RepositoryNotFoundError:
                pass
            else:
                raise SafetyError('Refusing an existing repository without this local ownership journal')
            self.state['phase'] = 'creating'
            self.save()
            self.api.create_repo(self.repo_id, repo_type='model', private=True, exist_ok=False)
            self.state['phase'] = 'created'
            self.save()
        if self.state['phase'] == 'creating':
            try:
                self.api.repo_info(self.repo_id, repo_type='model')
            except RepositoryNotFoundError:
                self.api.create_repo(self.repo_id, repo_type='model', private=True, exist_ok=False)
                self.state['phase'] = 'created'
                self.save()
            else:
                # Repo creation has no ownership token or compare-and-swap API.
                raise SafetyError('Ambiguous repo creation response; inspect/remove the empty repo before restarting with a new journal')
        if self.state['phase'] == 'created':
            head = self.refs()
            tree = self.tree(head)
            if set(tree) - {'.gitattributes'}:
                raise SafetyError('New repository unexpectedly contains foreign files')
            self.state['base_tree'] = tree
            self.state['head'] = head
            self.state['manifest'] = self.manifest(str(uuid.uuid4()))
            self.state['phase'] = 'claiming'
            self.save()
        if self.state['phase'] == 'claiming':
            head = self.refs()
            tree = self.tree(head)
            if head == self.state['head'] and tree == self.state['base_tree']:
                self.commit([], self.state['manifest'], head)
            head, _ = self.verified_head(self.state['manifest'], self.state['head'], allow_rewrite=True)
            self.state['head'] = head
            self.state['phase'] = 'ready'
            self.save()

    def candidates(self):
        entries = []
        for kind in ('full', 'eval'):
            directory = self.run_dir / 'export_queue' / kind
            matches = []
            seen_steps = set()
            for path in directory.glob('*.pt'):
                if path.is_symlink():
                    raise SafetyError(f'Queue checkpoint must not be a symlink: {path}')
                match = STEP.fullmatch(path.name)
                if not match:
                    raise SafetyError(f'Unexpected queue filename: {path}')
                step = int(match.group(1))
                if not 0 < step <= self.max_steps or (kind == 'eval' and step % self.eval_every):
                    raise SafetyError(f'Unexpected {kind} checkpoint step: {step}')
                if step in seen_steps:
                    raise SafetyError(f'Duplicate {kind} queue checkpoint step: {step}')
                seen_steps.add(step)
                matches.append((step, path))
            if kind == 'full' and matches:
                matches = [max(matches)]
            for step, path in sorted(matches):
                current = self.state['manifest']['full']
                if kind == 'full' and current and step < current['step']:
                    continue
                entries.append({'kind': kind, 'step': step, 'source': str(path),
                                'path': f'{"eval" if kind == "eval" else "resume"}/step-{step:08d}.pt'})
        return entries

    def prepare(self):
        entries = self.candidates()
        if not entries:
            return False
        transaction = str(uuid.uuid4())
        for index, entry in enumerate(entries):
            entry['stage'] = str(self.staging_dir / 'files' / f'{transaction}-{index}.pt')
        self.state['pending'] = {'id': transaction, 'phase': 'stage', 'entries': entries,
                                 'parent': self.state['head'], 'old_full': [], 'gc_intent': []}
        self.save()
        return True

    def stage(self, pending):
        kept = []
        for entry in pending['entries']:
            target = Path(entry['stage'])
            if not target.exists():
                try:
                    os.link(entry['source'], target)
                except FileNotFoundError:
                    if entry['kind'] == 'full':
                        continue
                    raise SafetyError(f'Eval queue disappeared before staging: {entry["source"]}')
                sync_dir(target.parent)
            with target.open('rb') as stream:
                os.fsync(stream.fileno())
            before = target.stat()
            hashes = digest(target)
            after = target.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise SafetyError('Queued checkpoint mutated while staging')
            entry.update(hashes)
            kept.append(entry)
        pending['entries'] = kept
        after = json.loads(json.dumps(self.state['manifest']))
        after['transaction'] = pending['id']
        for entry in kept:
            item = {key: entry[key] for key in ('step', 'path', 'size', 'sha256', 'git_oid')}
            old = (after['eval'].get(str(entry['step'])) if entry['kind'] == 'eval'
                   else after['full'])
            if old and old['step'] == entry['step'] and old != item:
                raise SafetyError(f'Immutable checkpoint changed: {entry["path"]}')
            if entry['kind'] == 'eval':
                after['eval'][str(entry['step'])] = item
            else:
                after['full'] = item
        _, tree = self.verified_head(self.state['manifest'], pending['parent'])
        old = self.state['manifest']['full']
        if old and old != after['full']:
            pending['old_full'] = [dict(old, lfs_oid=tree[old['path']]['sha256'])]
        pending['after'] = after
        pending['phase'] = 'commit'
        self.save()

    def publish(self, pending):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        head = self.refs()
        if head == pending['parent']:
            self.verified_head(self.state['manifest'], head)
            operations = []
            for entry in pending['entries']:
                if digest(Path(entry['stage'])) != {key: entry[key] for key in ('size', 'sha256', 'git_oid')}:
                    raise SafetyError('Staged checkpoint changed before upload')
                operations.append(CommitOperationAdd(path_in_repo=entry['path'], path_or_fileobj=entry['stage']))
            operations.extend(CommitOperationDelete(path_in_repo=old['path']) for old in pending['old_full'])
            self.commit(operations, pending['after'], pending['parent'])
        # A lost commit response is recoverable only with the exact transaction tree and title.
        head, _ = self.verified_head(pending['after'], pending['parent'], allow_rewrite=True)
        pending['committed_head'] = head
        pending['phase'] = 'committed'
        self.save()

    def collect(self, pending):
        manifest = pending['after']
        recovering = pending['phase'] == 'gc'
        head, tree = self.verified_head(manifest, pending['committed_head'], allow_rewrite=recovering)
        live = {item['sha256'] for item in tree.values() if item['sha256']}
        if not recovering:
            pending['gc_intent'] = [old for old in pending['old_full']
                                    if old['lfs_oid'] and old['lfs_oid'] not in live]
            pending['phase'] = 'gc'
            pending['pre_gc_head'] = head
            self.save()
        intent = {old['lfs_oid']: old for old in pending['gc_intent']}
        if set(intent) & live:
            raise SafetyError('Refusing GC of a currently live LFS object')
        listed = list(self.api.list_lfs_files(self.repo_id, repo_type='model')) if intent else []
        deletions = []
        for item in listed:
            if item.file_oid not in intent:
                continue
            old = intent[item.file_oid]
            if (item.filename != old['path'] or item.size != old['size']
                    or item.ref not in (None, 'main', 'refs/heads/main')):
                raise SafetyError('GC object does not match the exact recorded old-full path/size/ref')
            if not re.fullmatch(r'resume/step-\d{8}\.pt', item.filename):
                raise SafetyError('GC candidate is not a known full checkpoint')
            deletions.append(item)
        # Check refs, marker, complete live tree and head again immediately before destruction.
        self.verified_head(manifest, head)
        if deletions:
            self.status('garbage_collecting')
            self.api.permanently_delete_lfs_files(self.repo_id, deletions, repo_type='model', rewrite_history=True)
        remaining = {item.file_oid for item in self.api.list_lfs_files(self.repo_id, repo_type='model')} if intent else set()
        if remaining & set(intent):
            raise RetryableError('Old full LFS objects still listed after GC; preserving journal for retry')
        head, _ = self.verified_head(manifest, head, allow_rewrite=bool(intent))
        pending['finished_head'] = head
        pending['phase'] = 'ack'
        self.save()

    def acknowledge(self, pending):
        self.verified_head(pending['after'], pending['finished_head'])
        for entry in pending['entries']:
            source, stage = Path(entry['source']), Path(entry['stage'])
            try:
                if os.path.samefile(source, stage):
                    source.unlink()
                    sync_dir(source.parent)
            except FileNotFoundError:
                pass
        self.state['manifest'] = pending['after']
        self.state['head'] = pending['finished_head']
        for old in pending['gc_intent']:
            if old['lfs_oid'] not in self.state['gc_freed_oids']:
                self.state['gc_freed_oids'].append(old['lfs_oid'])
                self.state['gc_freed_bytes'] += old['size']
        self.state.setdefault('transactions', []).append({
            'id': pending['id'], 'parent': pending['parent'],
            'committed_head': pending['committed_head'], 'finished_head': pending['finished_head'],
            'old_full': pending['old_full'], 'gc_intent': pending['gc_intent'], 'verified_at': now()})
        self.state['last_success'] = now()
        self.state['last_error'] = None
        pending['phase'] = 'cleanup'
        self.save()

    def finish_pending(self):
        pending = self.state['pending']
        if not pending:
            return
        self.status('uploading')
        if pending['phase'] == 'stage':
            self.stage(pending)
        if pending['phase'] == 'commit':
            self.publish(pending)
        if pending['phase'] in ('committed', 'gc'):
            self.collect(pending)
        if pending['phase'] == 'ack':
            self.acknowledge(pending)
        if pending['phase'] == 'cleanup':
            for entry in pending['entries']:
                Path(entry['stage']).unlink(missing_ok=True)
            sync_dir(self.staging_dir / 'files')
            self.state['pending'] = None
            self.save()

    def run_once(self):
        self.authenticate()
        self.initialize()
        self.finish_pending()
        self.verified_head(self.state['manifest'], self.state['head'])
        if self.prepare():
            self.finish_pending()
        self.state['last_success'] = now()
        self.state['last_error'] = None
        self.save()
        expected = set(range(self.eval_every, self.max_steps + 1, self.eval_every))
        complete = (self.state['manifest']['full'] or {}).get('step') == self.max_steps
        complete = complete and expected == {int(step) for step in self.state['manifest']['eval']}
        self.status('complete' if complete else 'idle')
        return complete


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--run-dir', type=Path, required=True)
    result.add_argument('--staging-dir', type=Path, required=True)
    result.add_argument('--repo-id', required=True)
    result.add_argument('--run-id', help='Stable run identity; defaults to run-dir basename')
    result.add_argument('--max-steps', type=int, default=300000)
    result.add_argument('--eval-every', type=int, default=10000)
    result.add_argument('--poll-seconds', type=float, default=30)
    result.add_argument('--http-timeout', type=float, default=120)
    result.add_argument('--metadata', action='append', default=[], metavar='KEY=VALUE')
    result.add_argument('--metadata-json', type=Path)
    result.add_argument('--once', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.poll_seconds <= 0 or args.http_timeout <= 0:
        raise SystemExit('poll-seconds and http-timeout must be positive')
    metadata = json.loads(args.metadata_json.read_text()) if args.metadata_json else {}
    if not isinstance(metadata, dict):
        raise SystemExit('metadata-json must contain an object')
    for item in args.metadata:
        if '=' not in item:
            raise SystemExit('metadata must use KEY=VALUE')
        key, value = item.split('=', 1)
        metadata[key] = value
    api = make_api(args.http_timeout)
    try:
        with CheckpointUploader(args.run_dir, args.staging_dir, args.repo_id, api,
                                run_id=args.run_id, max_steps=args.max_steps,
                                eval_every=args.eval_every, metadata=metadata) as uploader:
            while True:
                try:
                    complete = uploader.run_once()
                except SafetyError as exc:
                    uploader.record_error(exc)
                    print(json.dumps({'event': 'fatal', 'error': redact_credentials(exc)}), flush=True)
                    return 2
                except Exception as exc:
                    uploader.record_error(exc)
                    print(json.dumps({'event': 'retrying', 'error': redact_credentials(exc)}), flush=True)
                    if args.once:
                        return 1
                else:
                    print(json.dumps({'event': 'healthy', 'step': (uploader.state['manifest']['full'] or {}).get('step', 0),
                                      'gc_freed_bytes': uploader.state['gc_freed_bytes']}), flush=True)
                    if args.once or complete:
                        return 0
                time.sleep(args.poll_seconds)
    except SafetyError as exc:
        print(json.dumps({'event': 'fatal', 'error': redact_credentials(exc)}), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
