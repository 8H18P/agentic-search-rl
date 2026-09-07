"""Shared IO primitives for hardened teacher-rollout runs.

Design goals (Data-Pipeline Hardening):
  * run-dir isolation        -> each rollout gets logs/teacher_rollout/<run_id>/
  * single-writer lock       -> O_EXCL lock file prevents two processes writing one run dir
  * events are authoritative -> per-run events.jsonl (RunObserver injects question_id)
  * append-only result writer with per-line fsync
  * atomic completed_ids ledger (tmp + os.replace) for resume / no-duplicate guarantee
  * results.jsonl tail repair on resume after a crash

No Champion-core files are touched.
"""
from __future__ import annotations
import copy, json, os, time
from pathlib import Path


def create_run_dir(run_dir, resume: bool = False) -> Path:
    """Create/validate a run directory and take the single-writer lock."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / ".writer.lock"
    if lock.exists():
        if not resume:
            raise RuntimeError(
                f"single-writer lock exists: {lock}. Another process is active or a prior run "
                f"crashed. Verify no active process, then re-run with --resume."
            )
        lock.unlink(missing_ok=True)  # resume takeover (caller verified no active writer)
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "started": time.time(),
                                 "host": os.uname().nodename}).encode("utf-8"))
    finally:
        os.close(fd)
    return run_dir


def release_lock(run_dir) -> None:
    (Path(run_dir) / ".writer.lock").unlink(missing_ok=True)


class RunObserver:
    """Per-run observability sidecar. Writes to <run_dir>/events.jsonl and injects
    question_id into every event so events + manifest fully reconstruct a rollout."""

    def __init__(self, path, trajectory_id: str, question_id: str):
        self.path = Path(path)
        self.trajectory_id = trajectory_id
        self.question_id = str(question_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def record_event(self, event_type: str, payload=None) -> None:
        try:
            item = {
                "event_type": event_type,
                "trajectory_id": self.trajectory_id,
                "question_id": self.question_id,
                "timestamp": time.time(),
                "payload": copy.deepcopy(payload or {}),
            }
            self._fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
        except Exception:
            pass


class ResultWriter:
    """Append-only JSONL writer for derived per-question results (fsync per line)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def append(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
        except Exception:
            pass


def load_completed(run_dir) -> set:
    p = Path(run_dir) / "completed_ids.json"
    if not p.exists():
        return set()
    try:
        return set(json.load(open(p, encoding="utf-8")))
    except Exception:
        return set()


def update_completed(run_dir, qid) -> None:
    p = Path(run_dir) / "completed_ids.json"
    ids = load_completed(run_dir)
    ids.add(str(qid))
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sorted(ids), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)  # atomic


def repair_results_tail(path) -> None:
    """Drop a partial trailing JSONL line left by a mid-write crash."""
    p = Path(path)
    if not p.exists():
        return
    data = p.read_bytes()
    nl = data.rfind(b"\n")
    if nl == -1:
        p.write_bytes(b"")
        return
    if data[nl + 1:].strip():
        p.write_bytes(data[:nl + 1])
