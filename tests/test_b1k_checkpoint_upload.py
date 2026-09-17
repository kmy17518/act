"""Single-writer publication, crash recovery and opt-in live LFS collection tests."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.errors import RepositoryNotFoundError


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'b1k_checkpoint_upload.py'
if not MODULE.exists():
    MODULE = ROOT / 'diffusion_policy/b1k/checkpoint_upload.py'
spec = importlib.util.spec_from_file_location('checkpoint_upload_under_test', MODULE)
upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)


class FakeApi:
    def __init__(self):
        self.exists = False
        self.head = '0' * 40
        self.title = 'initial commit'
        self.files = {'.gitattributes': b'*.pt filter=lfs diff=lfs merge=lfs -text\n'}
        self.lfs = {}
        self.deleted = []
        self.commits = []
        self.tags = []
        self.prs = []
        self.branches = []
        self.fail_commit = False
        self.fail_gc = False
        self.fail_before_gc = False
        self.fail_before_commit = False
        self.gc_leaves_listed = False

    def whoami(self):
        return {'name': 'tester'}

    def repo_info(self, *args, **kwargs):
        if not self.exists:
            raise RepositoryNotFoundError('missing')
        return SimpleNamespace(sha=self.head)

    def create_repo(self, *args, **kwargs):
        assert kwargs['private'] and not kwargs['exist_ok']
        assert not self.exists
        self.exists = True

    def list_repo_refs(self, *args, **kwargs):
        assert kwargs['include_pull_requests']
        return SimpleNamespace(branches=[SimpleNamespace(ref='refs/heads/main', target_commit=self.head)] + self.branches,
                               tags=self.tags, pull_requests=self.prs, converts=[])

    def list_repo_tree(self, *args, **kwargs):
        assert kwargs['revision'] == self.head
        for path, data in sorted(self.files.items()):
            lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if path.endswith('.pt') else None
            yield SimpleNamespace(path=path, size=len(data), blob_id=upload.git_oid(data), lfs=lfs)

    def advance(self):
        self.head = f'{int(self.head, 16) + 1:040x}'

    def create_commit(self, *args, **kwargs):
        assert kwargs['parent_commit'] == self.head
        if self.fail_before_commit:
            self.fail_before_commit = False
            raise ConnectionError('connection failed before commit')
        self.commits.append(kwargs)
        for operation in kwargs['operations']:
            if isinstance(operation, CommitOperationAdd):
                value = operation.path_or_fileobj
                data = value if isinstance(value, bytes) else Path(value).read_bytes()
                self.files[operation.path_in_repo] = data
                if operation.path_in_repo.endswith('.pt'):
                    oid = hashlib.sha256(data).hexdigest()
                    self.lfs.setdefault(oid, SimpleNamespace(file_oid=oid, filename=operation.path_in_repo,
                                                            size=len(data), ref='refs/heads/main'))
            elif isinstance(operation, CommitOperationDelete):
                del self.files[operation.path_in_repo]
            else:
                raise AssertionError(type(operation))
        self.advance()
        self.title = kwargs['commit_message']
        if self.fail_commit:
            self.fail_commit = False
            raise ConnectionError('lost commit response')
        return SimpleNamespace(oid=self.head)

    def list_repo_commits(self, *args, **kwargs):
        return [SimpleNamespace(commit_id=self.head, title=self.title)]

    def list_lfs_files(self, *args, **kwargs):
        return list(self.lfs.values())

    def permanently_delete_lfs_files(self, repo_id, files, **kwargs):
        assert kwargs['rewrite_history'] is True
        if self.fail_before_gc:
            self.fail_before_gc = False
            raise ConnectionError('connection failed before GC')
        live = {hashlib.sha256(data).hexdigest() for path, data in self.files.items() if path.endswith('.pt')}
        for item in files:
            assert item.file_oid not in live
            self.deleted.append(item.file_oid)
            if not self.gc_leaves_listed:
                del self.lfs[item.file_oid]
        self.advance()
        if self.fail_gc:
            self.fail_gc = False
            raise ConnectionError('lost GC response after history rewrite')


def queue(run, kind, step, data=b'checkpoint', separator='_'):
    directory = run / 'export_queue' / kind
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'step{separator}{step:08d}.pt'
    path.write_bytes(data)
    return path


def uploader(tmp_path, api, **kwargs):
    return upload.CheckpointUploader(tmp_path / 'run', tmp_path / 'stage', 'tester/dedicated', api, **kwargs)


def test_two_full_replacements_keep_eval_and_collect_only_recorded_oid(tmp_path):
    api = FakeApi()
    run = tmp_path / 'run'
    first = queue(run, 'full', 2500, b'first full')
    evaluation = queue(run, 'eval', 10000, b'evaluation', separator='-')
    with uploader(tmp_path, api) as worker:
        worker.run_once()
        old_oid = hashlib.sha256(b'first full').hexdigest()
        assert not first.exists() and not evaluation.exists()
        unrelated = SimpleNamespace(file_oid='unknown', filename='resume/step-00000001.pt', size=100, ref='main')
        api.lfs['unknown'] = unrelated
        queue(run, 'full', 5000, b'second full', separator='-')
        worker.run_once()
        assert api.deleted == [old_oid]
        assert api.files['eval/step-00010000.pt'] == b'evaluation'
        assert [path for path in api.files if path.startswith('resume/')] == ['resume/step-00005000.pt']
        assert 'unknown' in api.lfs
        assert not list((tmp_path / 'stage/files').iterdir())
        status = json.loads(worker.status_path.read_text())
        assert status['current_step'] == 5000
        assert status['gc_freed_bytes'] == len(b'first full')
        assert status['last_success'] and status['health'] == 'idle'
        replacement = api.commits[-1]['operations']
        assert any(isinstance(op, CommitOperationDelete) for op in replacement)
        assert any(isinstance(op, CommitOperationAdd) and op.path_in_repo == 'resume/step-00005000.pt' for op in replacement)


@pytest.mark.parametrize('failure', ['fail_commit', 'fail_gc', 'fail_before_commit', 'fail_before_gc'])
def test_crash_and_lost_responses_resume_durable_transaction(tmp_path, failure):
    api = FakeApi()
    run = tmp_path / 'run'
    queue(run, 'full', 2500, b'old')
    with uploader(tmp_path, api) as worker:
        worker.run_once()
    queue(run, 'full', 5000, b'new')
    evaluation = queue(run, 'eval', 10000, b'eval')
    setattr(api, failure, True)
    with uploader(tmp_path, api) as worker:
        with pytest.raises(ConnectionError):
            worker.run_once()
        assert worker.state['pending']
        assert evaluation.exists()
        assert list((tmp_path / 'stage/files').iterdir())
    # Simulate trainer retention removing the full queue after staging.
    for path in (run / 'export_queue/full').glob('*.pt'):
        path.unlink()
    with uploader(tmp_path, api) as worker:
        worker.run_once()
        assert worker.state['pending'] is None
        assert worker.state['gc_freed_bytes'] == len(b'old')
        assert not evaluation.exists()
        assert api.files['resume/step-00005000.pt'] == b'new'
        assert api.files['eval/step-00010000.pt'] == b'eval'
        assert len(api.deleted) == 1


@pytest.mark.parametrize('where', ['tags', 'prs', 'branches'])
def test_foreign_refs_fail_closed_before_upload_or_gc(tmp_path, where):
    api = FakeApi()
    queue(tmp_path / 'run', 'full', 2500, b'old')
    with uploader(tmp_path, api) as worker:
        worker.run_once()
        queue(tmp_path / 'run', 'full', 5000, b'new')
        getattr(api, where).append(SimpleNamespace(ref='foreign', target_commit=api.head))
        with pytest.raises(upload.SafetyError, match='only main'):
            worker.run_once()
        assert not api.deleted
        assert 'resume/step-00002500.pt' in api.files


@pytest.mark.parametrize('change', ['marker', 'head', 'extra_path'])
def test_foreign_writer_fail_closed(tmp_path, change):
    api = FakeApi()
    with uploader(tmp_path, api) as worker:
        worker.run_once()
        if change == 'marker':
            api.files[upload.MARKER] = b'foreign owner'
        elif change == 'head':
            api.advance()
        else:
            api.files['foreign.bin'] = b'foreign'
        with pytest.raises(upload.SafetyError):
            worker.run_once()
        assert not api.deleted


def test_existing_unowned_repo_rejected(tmp_path):
    api = FakeApi()
    api.exists = True
    with uploader(tmp_path, api) as worker:
        with pytest.raises(upload.SafetyError, match='existing repository'):
            worker.run_once()
    assert not api.commits


def test_local_lock_covers_run_and_staging(tmp_path):
    api = FakeApi()
    with uploader(tmp_path, api):
        with pytest.raises(upload.SafetyError, match='Another uploader'):
            with uploader(tmp_path, api):
                pass
        with pytest.raises(upload.SafetyError, match='Another uploader'):
            with upload.CheckpointUploader(tmp_path / 'run', tmp_path / 'stage2', 'tester/dedicated', api):
                pass


def test_live_eval_oid_never_collected_even_when_shared_with_old_full(tmp_path):
    api = FakeApi()
    run = tmp_path / 'run'
    queue(run, 'full', 2500, b'shared')
    queue(run, 'eval', 10000, b'shared')
    with uploader(tmp_path, api) as worker:
        worker.run_once()
        queue(run, 'full', 5000, b'new')
        worker.run_once()
        assert not api.deleted
        assert worker.state['gc_freed_bytes'] == 0
        assert api.files['eval/step-00010000.pt'] == b'shared'


def test_gc_filename_and_size_must_match_recorded_full(tmp_path):
    api = FakeApi()
    run = tmp_path / 'run'
    queue(run, 'full', 2500, b'old')
    with uploader(tmp_path, api) as worker:
        worker.run_once()
        api.lfs[hashlib.sha256(b'old').hexdigest()].filename = 'eval/foreign.pt'
        queue(run, 'full', 5000, b'new')
        with pytest.raises(upload.SafetyError, match='exact recorded'):
            worker.run_once()
        assert not api.deleted
        assert worker.state['pending']['phase'] == 'gc'


def test_unverified_gc_keeps_eval_queue_and_journal(tmp_path):
    api = FakeApi()
    run = tmp_path / 'run'
    queue(run, 'full', 2500, b'old')
    with uploader(tmp_path, api) as worker:
        worker.run_once()
        queue(run, 'full', 5000, b'new')
        evaluation = queue(run, 'eval', 10000, b'eval')
        api.gc_leaves_listed = True
        with pytest.raises(upload.RetryableError):
            worker.run_once()
        assert evaluation.exists()
        assert worker.state['pending']['gc_intent'][0]['lfs_oid'] == hashlib.sha256(b'old').hexdigest()
        api.gc_leaves_listed = False
        worker.run_once()
        assert not evaluation.exists()


def test_changed_eval_is_rejected(tmp_path):
    api = FakeApi()
    run = tmp_path / 'run'
    queue(run, 'eval', 10000, b'first')
    with uploader(tmp_path, api) as worker:
        worker.run_once()
        queue(run, 'eval', 10000, b'changed')
        with pytest.raises(upload.SafetyError, match='Immutable'):
            worker.run_once()
        assert api.files['eval/step-00010000.pt'] == b'first'


def test_ack_crash_recovers_without_duplicate_gc_accounting(tmp_path):
    api = FakeApi()
    run = tmp_path / 'run'
    queue(run, 'full', 2500, b'old')
    with uploader(tmp_path, api) as worker:
        worker.run_once()
        queue(run, 'full', 5000, b'new')
        worker.prepare()
        pending = worker.state['pending']
        worker.stage(pending)
        worker.publish(pending)
        worker.collect(pending)
        worker.acknowledge(pending)
    with uploader(tmp_path, api) as worker:
        worker.run_once()
        assert worker.state['gc_freed_bytes'] == 3
        assert len(api.deleted) == 1
        assert not list((tmp_path / 'stage/files').iterdir())


def test_full_step_one_accepted_and_completion_requires_every_eval(tmp_path):
    api = FakeApi()
    run = tmp_path / 'run'
    queue(run, 'full', 1, b'initial')
    with uploader(tmp_path, api, max_steps=20000) as worker:
        assert not worker.run_once()
        queue(run, 'full', 20000, b'final')
        queue(run, 'eval', 20000, b'eval final')
        assert not worker.run_once()
        queue(run, 'eval', 10000, b'eval middle')
        assert worker.run_once()
        assert json.loads(worker.status_path.read_text())['health'] == 'complete'


def test_wrong_account_rejected(tmp_path):
    api = FakeApi()
    api.whoami = lambda: {'name': 'foreign'}
    with uploader(tmp_path, api) as worker:
        with pytest.raises(upload.SafetyError, match='Authenticated'):
            worker.run_once()
    assert not api.exists


def test_backend_has_finite_timeouts(monkeypatch):
    import requests
    from huggingface_hub import constants, get_session
    seen = []
    monkeypatch.setattr(requests.Session, 'request', lambda self, *args, **kwargs: seen.append(kwargs))
    upload.make_api(77)
    get_session().get('https://example.invalid')
    assert seen[0]['timeout'] == (10, 77)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert constants.HF_HUB_DISABLE_XET is True
    assert constants.HF_HUB_ENABLE_HF_TRANSFER is False


@pytest.mark.parametrize('prefix', ['https://example.invalid/object', '/object'])
def test_signed_url_and_token_redaction(prefix, monkeypatch):
    monkeypatch.setenv('HF_TOKEN', 'private-test-token')
    message = f'SSLError: {prefix}?X-Amz-Credential=key&X-Amz-Signature=signature private-test-token'
    safe = upload.redact_credentials(message)
    assert safe == f'SSLError: {prefix}?[REDACTED] [REDACTED]'
    assert upload.redact_credentials('Authorization: Bearer test-secret') == 'Authorization: Bearer [REDACTED]'


def test_logging_filter_redacts_formatted_args_and_traceback():
    import logging
    import sys
    try:
        raise ConnectionError('/object?X-Amz-Signature=signature')
    except ConnectionError:
        record = logging.LogRecord('huggingface_hub.utils._http', logging.WARNING, __file__, 1,
                                   'retrying %s', ('/object?X-Amz-Signature=signature',), sys.exc_info())
    assert upload.CredentialFilter().filter(record)
    text = logging.Formatter().format(record)
    assert 'signature' not in text and 'ConnectionError' in text and '[REDACTED]' in text


def test_cli_retry_redacts_output_and_journal(tmp_path, monkeypatch, capsys):
    api = FakeApi()
    def fail():
        raise ConnectionError('/object?X-Amz-Signature=signature')
    api.whoami = fail
    monkeypatch.setattr(upload, 'make_api', lambda *args: api)
    assert upload.main(['--run-dir', str(tmp_path / 'run'), '--staging-dir', str(tmp_path / 'stage'),
                        '--repo-id', 'tester/dedicated', '--once']) == 1
    for text in [capsys.readouterr().out, (tmp_path / 'stage/status.json').read_text(),
                 (tmp_path / 'stage/state.json').read_text()]:
        assert 'signature' not in text and '[REDACTED]' in text


def test_repo_creation_network_failure_before_creation_retries(tmp_path):
    api = FakeApi()
    original = api.create_repo
    def fail(*args, **kwargs):
        raise ConnectionError('failed before creation')
    api.create_repo = fail
    with uploader(tmp_path, api) as worker:
        with pytest.raises(ConnectionError):
            worker.run_once()
        assert worker.state['phase'] == 'creating'
    api.create_repo = original
    with uploader(tmp_path, api) as worker:
        worker.run_once()
        assert worker.state['phase'] == 'ready'


def test_ambiguous_repo_creation_fails_closed(tmp_path):
    api = FakeApi()
    original = api.create_repo
    def lose_response(*args, **kwargs):
        original(*args, **kwargs)
        raise ConnectionError('lost creation response')
    api.create_repo = lose_response
    with uploader(tmp_path, api) as worker:
        with pytest.raises(ConnectionError):
            worker.run_once()
    with uploader(tmp_path, api) as worker:
        with pytest.raises(upload.SafetyError, match='Ambiguous'):
            worker.run_once()
    assert not api.commits


def test_lost_ownership_marker_commit_response_recovers(tmp_path):
    api = FakeApi()
    api.fail_commit = True
    with uploader(tmp_path, api) as worker:
        with pytest.raises(ConnectionError):
            worker.run_once()
    with uploader(tmp_path, api) as worker:
        worker.run_once()
        assert worker.state['phase'] == 'ready'
    assert len(api.commits) == 1


def test_pr_appearing_during_gc_listing_blocks_deletion(tmp_path):
    api = FakeApi()
    run = tmp_path / 'run'
    queue(run, 'full', 2500, b'old')
    with uploader(tmp_path, api) as worker:
        worker.run_once()
        queue(run, 'full', 5000, b'new')
        original = api.list_lfs_files
        def add_pr(*args, **kwargs):
            api.prs.append(SimpleNamespace(ref='refs/pr/1', target_commit=api.head))
            return original(*args, **kwargs)
        api.list_lfs_files = add_pr
        with pytest.raises(upload.SafetyError, match='only main'):
            worker.run_once()
        assert not api.deleted


def test_duplicate_step_filenames_rejected(tmp_path):
    api = FakeApi()
    run = tmp_path / 'run'
    queue(run, 'eval', 10000, b'eval1', separator='_')
    queue(run, 'eval', 10000, b'eval2', separator='-')
    with uploader(tmp_path, api) as worker:
        with pytest.raises(upload.SafetyError, match='Duplicate'):
            worker.run_once()


def test_cli_exits_when_all_checkpoints_complete(tmp_path, monkeypatch):
    api = FakeApi()
    run = tmp_path / 'run'
    queue(run, 'full', 10000, b'full')
    queue(run, 'eval', 10000, b'eval')
    monkeypatch.setattr(upload, 'make_api', lambda *args: api)
    assert upload.main(['--run-dir', str(run), '--staging-dir', str(tmp_path / 'stage'),
                        '--repo-id', 'tester/dedicated', '--max-steps', '10000']) == 0


def test_cli_fatal_status_and_exit_code(tmp_path, monkeypatch):
    api = FakeApi()
    api.exists = True
    monkeypatch.setattr(upload, 'make_api', lambda *args: api)
    assert upload.main(['--run-dir', str(tmp_path / 'run'), '--staging-dir', str(tmp_path / 'stage'),
                        '--repo-id', 'tester/dedicated', '--once']) == 2
    status = json.loads((tmp_path / 'stage/status.json').read_text())
    assert status['health'] == 'fatal' and status['error_count'] == 1


@pytest.mark.skipif(os.environ.get('B1K_LIVE_UPLOAD_TEST') != '1', reason='explicit private HF smoke opt-in required')
def test_live_private_repo_replacement_and_permanent_gc(tmp_path):
    from huggingface_hub import hf_hub_download
    api = upload.make_api()
    assert api.whoami()['name'] == 'kmy17518'
    repo_id = 'kmy17518/b1k-upload-smoke-' + uuid.uuid4().hex[:16]
    run = tmp_path / 'live-run'
    stage = tmp_path / 'live-stage'
    first_bytes = os.urandom(12 * 1024 * 1024)
    second_bytes = os.urandom(12 * 1024 * 1024)
    eval_bytes = os.urandom(11 * 1024 * 1024)
    old_oid = hashlib.sha256(first_bytes).hexdigest()
    new_oid = hashlib.sha256(second_bytes).hexdigest()
    eval_oid = hashlib.sha256(eval_bytes).hexdigest()
    queue(run, 'full', 2500, first_bytes)
    queue(run, 'eval', 10000, eval_bytes)
    evidence = {'repo_id': repo_id, 'old_oid': old_oid, 'new_oid': new_oid, 'eval_oid': eval_oid,
                'full_size_bytes': len(first_bytes), 'eval_size_bytes': len(eval_bytes)}
    created = False
    try:
        with upload.CheckpointUploader(run, stage, repo_id, api, run_id='private-live-smoke') as worker:
            worker.run_once()
            created = True
            assert api.repo_info(repo_id).private
            assert old_oid in {item.file_oid for item in api.list_lfs_files(repo_id)}
            evidence['first_head'] = worker.state['head']
            queue(run, 'full', 5000, second_bytes, separator='-')
            worker.run_once()
            listed = {item.file_oid for item in api.list_lfs_files(repo_id)}
            assert old_oid not in listed
            assert new_oid in listed and eval_oid in listed
            paths = api.list_repo_files(repo_id)
            assert [path for path in paths if path.startswith('resume/')] == ['resume/step-00005000.pt']
            assert 'eval/step-00010000.pt' in paths
            downloaded = hf_hub_download(repo_id, 'eval/step-00010000.pt', revision=worker.state['head'],
                                         local_dir=tmp_path / 'verified-eval', token=os.environ.get('HF_TOKEN'))
            assert hashlib.sha256(Path(downloaded).read_bytes()).hexdigest() == eval_oid
            assert not list((run / 'export_queue/eval').glob('*.pt'))
            evidence.update({'final_head': worker.state['head'], 'remaining_lfs_oids': sorted(listed),
                             'remote_paths': paths, 'gc_freed_bytes': worker.state['gc_freed_bytes'],
                             'eval_download_verified': True, 'old_oid_absent': True})
            assert worker.state['gc_freed_bytes'] == len(first_bytes)
    finally:
        # The random private repository is the only remote resource this test may delete.
        if created:
            api.delete_repo(repo_id, repo_type='model')
            evidence['scratch_repo_deleted'] = True
        print('B1K_LIVE_EVIDENCE ' + json.dumps(evidence, sort_keys=True), flush=True)
