"""The job queue behind the API: one PDF at a time, in a background thread.

A PDF takes about 30 seconds, longer than a web request should stay open, so an
upload returns a job id at once and the browser asks for the job's status every few
seconds. One worker runs the jobs in order: the model server has one GPU, so two runs
at once would only slow each other down.

Jobs live in memory. Each upload is saved in its own temporary folder, which is
deleted when the job finishes; a finished job is forgotten ``ttl`` seconds later.
Restarting the server forgets every job, and asking for a forgotten id gives ``None``.
"""
import queue
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..utils import get_logger

logger = get_logger(__name__)

# What a running job is doing, in order. The web app shows each one while it waits.
STAGES = ("reading", "narrative", "rda")

# Takes the PDF and a function to report the current stage; returns the merged JSON.
Runner = Callable[[Path, Callable[[str], None]], dict]


class JobFailed(Exception):
    """A failure whose message is written for the user and shown as it is."""


class QueueFull(Exception):
    """Too many jobs are already waiting."""


@dataclass
class Job:
    id: str
    folder: Path
    status: str = "queued"              # queued | running | done | failed
    stage: Optional[str] = None         # one of STAGES while running
    result: Optional[dict] = None
    error: Optional[str] = None
    finished: Optional[float] = None    # time.monotonic() when done or failed

    @property
    def pdf_path(self) -> Path:
        return self.folder / "upload.pdf"


@dataclass
class JobQueue:
    run: Runner
    ttl: float = 3600.0
    max_waiting: int = 20
    _jobs: dict = field(default_factory=dict)
    _waiting: queue.Queue = field(default_factory=queue.Queue)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> None:
        """Start the worker thread. It stops with the process."""
        threading.Thread(target=self._work, name="dmpbridge-jobs", daemon=True).start()

    def submit(self, pdf: bytes) -> Job:
        """Save *pdf* in a new folder and queue it. Raises :class:`QueueFull`."""
        with self._lock:
            self._forget_old()
            if self._waiting.qsize() >= self.max_waiting:
                raise QueueFull()
            job = Job(id=uuid.uuid4().hex, folder=Path(tempfile.mkdtemp(prefix="dmpbridge-")))
            job.pdf_path.write_bytes(pdf)
            self._jobs[job.id] = job
            self._waiting.put(job.id)
        logger.info("[api] job %s queued", job.id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            self._forget_old()
            return self._jobs.get(job_id)

    def position(self, job: Job) -> int:
        """How many jobs run before this one: 0 while it runs or once it is done."""
        if job.status != "queued":
            return 0
        with self._lock:
            ahead = [j for j in self._jobs.values() if j.status in ("queued", "running")]
        # Dicts keep insertion order, so the jobs before this one were submitted first.
        return [j.id for j in ahead].index(job.id)

    def waiting(self) -> int:
        return self._waiting.qsize()

    def _forget_old(self) -> None:
        """Drop finished jobs older than the ttl. Called with the lock held."""
        now = time.monotonic()
        for job_id in [j.id for j in self._jobs.values()
                       if j.finished is not None and now - j.finished > self.ttl]:
            del self._jobs[job_id]

    def _work(self) -> None:
        while True:
            job = self._jobs.get(self._waiting.get())
            if job is not None:
                self._run_one(job)

    def _run_one(self, job: Job) -> None:
        started = time.monotonic()
        job.status = "running"

        def set_stage(stage: str) -> None:
            job.stage = stage

        try:
            job.result = self.run(job.pdf_path, set_stage)
            job.status = "done"
        except JobFailed as e:
            job.error, job.status = str(e), "failed"
        except Exception:
            logger.exception("[api] job %s failed", job.id)
            job.error, job.status = "Something went wrong while reading this PDF. Please try again.", "failed"
        finally:
            job.stage, job.finished = None, time.monotonic()
            shutil.rmtree(job.folder, ignore_errors=True)
        logger.info("[api] job %s %s in %.1f s", job.id, job.status, job.finished - started)
