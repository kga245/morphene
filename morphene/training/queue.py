"""
Morphene Training Queue Manager

Manages parallel training jobs with:
- GPU memory monitoring
- Job scheduling and launching
- Central logging dashboard
- Failure alerts and auto-retry
"""

import os
import sys
import json
import time
import signal
import sqlite3
import subprocess
import threading
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a training job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class TrainingJob:
    """
    Represents a single training job.

    Attributes:
        id: Unique job identifier
        instrument_number: Instrument being trained
        instrument_name: Name of the instrument
        config: RAVE config to use
        db_path: Path to preprocessed dataset
        output_path: Path for training outputs
        status: Current job status
        gpu_id: GPU to use (or None for auto)
        max_steps: Maximum training steps
        batch_size: Training batch size
        priority: Job priority (higher = more important)
        retry_count: Number of retries attempted
        max_retries: Maximum retry attempts
        error_message: Last error message if failed
        pid: Process ID when running
        peak_memory_gb: Peak GPU memory usage
        started_at: When job started
        completed_at: When job completed
        created_at: When job was created
    """
    instrument_number: int
    instrument_name: str
    config: str = "v2"
    db_path: str = ""
    output_path: str = ""
    status: JobStatus = JobStatus.PENDING
    id: Optional[int] = None
    gpu_id: Optional[int] = None
    max_steps: int = 3_000_000
    batch_size: int = 8
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    error_message: str = ""
    pid: Optional[int] = None
    peak_memory_gb: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['status'] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingJob':
        """Create from dictionary."""
        d = d.copy()
        d['status'] = JobStatus(d['status'])
        return cls(**d)


class GPUMonitor:
    """Monitors GPU memory usage."""

    def __init__(self, gpu_ids: Optional[List[int]] = None):
        """
        Initialize GPU monitor.

        Args:
            gpu_ids: List of GPU IDs to monitor (None = all)
        """
        self.gpu_ids = gpu_ids

    def get_available_memory_gb(self, gpu_id: int = 0) -> float:
        """
        Get available GPU memory in GB.

        Args:
            gpu_id: GPU device ID

        Returns:
            Available memory in GB
        """
        if not HAS_GPUTIL:
            logger.warning("GPUtil not installed, assuming 128GB available")
            return 128.0

        try:
            gpus = GPUtil.getGPUs()
            if gpu_id < len(gpus):
                gpu = gpus[gpu_id]
                return (gpu.memoryTotal - gpu.memoryUsed) / 1024
            return 0.0
        except Exception as e:
            logger.error(f"Error getting GPU memory: {e}")
            return 0.0

    def get_used_memory_gb(self, gpu_id: int = 0) -> float:
        """Get used GPU memory in GB."""
        if not HAS_GPUTIL:
            return 0.0

        try:
            gpus = GPUtil.getGPUs()
            if gpu_id < len(gpus):
                return gpus[gpu_id].memoryUsed / 1024
            return 0.0
        except Exception as e:
            logger.error(f"Error getting GPU memory: {e}")
            return 0.0

    def get_total_memory_gb(self, gpu_id: int = 0) -> float:
        """Get total GPU memory in GB."""
        if not HAS_GPUTIL:
            return 128.0

        try:
            gpus = GPUtil.getGPUs()
            if gpu_id < len(gpus):
                return gpus[gpu_id].memoryTotal / 1024
            return 0.0
        except Exception as e:
            logger.error(f"Error getting GPU memory: {e}")
            return 0.0

    def can_launch_job(
        self,
        gpu_id: int,
        estimated_memory_gb: float,
        safety_margin_gb: float = 4.0
    ) -> bool:
        """
        Check if a job can be launched based on available memory.

        Args:
            gpu_id: GPU to check
            estimated_memory_gb: Estimated memory requirement
            safety_margin_gb: Safety margin to leave free

        Returns:
            True if job can be launched
        """
        available = self.get_available_memory_gb(gpu_id)
        return available >= (estimated_memory_gb + safety_margin_gb)


class TrainingQueue:
    """
    Manages a queue of training jobs.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instrument_number INTEGER NOT NULL,
        instrument_name TEXT NOT NULL,
        config TEXT DEFAULT 'v2',
        db_path TEXT NOT NULL,
        output_path TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        gpu_id INTEGER,
        max_steps INTEGER DEFAULT 3000000,
        batch_size INTEGER DEFAULT 8,
        priority INTEGER DEFAULT 0,
        retry_count INTEGER DEFAULT 0,
        max_retries INTEGER DEFAULT 3,
        error_message TEXT DEFAULT '',
        pid INTEGER,
        peak_memory_gb REAL DEFAULT 0.0,
        started_at TEXT,
        completed_at TEXT,
        created_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_status ON jobs(status);
    CREATE INDEX IF NOT EXISTS idx_priority ON jobs(priority DESC);

    CREATE TABLE IF NOT EXISTS job_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        level TEXT NOT NULL,
        message TEXT NOT NULL,
        FOREIGN KEY (job_id) REFERENCES jobs(id)
    );

    CREATE TABLE IF NOT EXISTS memory_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        gpu_id INTEGER NOT NULL,
        memory_used_gb REAL NOT NULL,
        FOREIGN KEY (job_id) REFERENCES jobs(id)
    );
    """

    def __init__(
        self,
        db_path: str,
        gpu_ids: Optional[List[int]] = None,
        max_parallel_jobs: int = 3,
        memory_per_job_gb: float = 32.0,
        on_job_complete: Optional[Callable[[TrainingJob], None]] = None,
        on_job_failed: Optional[Callable[[TrainingJob], None]] = None,
    ):
        """
        Initialize the training queue.

        Args:
            db_path: Path to SQLite database
            gpu_ids: GPU IDs to use (None = use GPU 0)
            max_parallel_jobs: Maximum parallel jobs
            memory_per_job_gb: Estimated memory per job
            on_job_complete: Callback when job completes
            on_job_failed: Callback when job fails
        """
        self.db_path = db_path
        self.gpu_ids = gpu_ids or [0]
        self.max_parallel_jobs = max_parallel_jobs
        self.memory_per_job_gb = memory_per_job_gb
        self.on_job_complete = on_job_complete
        self.on_job_failed = on_job_failed

        self.gpu_monitor = GPUMonitor(gpu_ids)
        self.running_jobs: Dict[int, subprocess.Popen] = {}
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_job(self, job: TrainingJob) -> int:
        """
        Add a job to the queue.

        Args:
            job: TrainingJob to add

        Returns:
            Job ID
        """
        job.created_at = datetime.now().isoformat()
        job.status = JobStatus.PENDING

        data = job.to_dict()
        del data['id']

        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"INSERT INTO jobs ({columns}) VALUES ({placeholders})",
                list(data.values())
            )
            job_id = cursor.lastrowid
            self._log_job(conn, job_id, "INFO", f"Job created for {job.instrument_name}")
            return job_id

    def get_job(self, job_id: int) -> Optional[TrainingJob]:
        """Get a job by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (job_id,)
            ).fetchone()
            if row:
                return TrainingJob.from_dict(dict(row))
        return None

    def update_job(self, job: TrainingJob) -> None:
        """Update a job."""
        if job.id is None:
            raise ValueError("Job must have an id to update")

        data = job.to_dict()
        del data['id']

        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE jobs SET {set_clause} WHERE id = ?",
                list(data.values()) + [job.id]
            )

    def get_pending_jobs(self) -> List[TrainingJob]:
        """Get all pending jobs ordered by priority."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = 'pending' ORDER BY priority DESC, id ASC"
            ).fetchall()
            return [TrainingJob.from_dict(dict(row)) for row in rows]

    def get_running_jobs(self) -> List[TrainingJob]:
        """Get all running jobs."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status = 'running'"
            ).fetchall()
            return [TrainingJob.from_dict(dict(row)) for row in rows]

    def get_all_jobs(self) -> List[TrainingJob]:
        """Get all jobs."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
            return [TrainingJob.from_dict(dict(row)) for row in rows]

    def _log_job(
        self,
        conn: sqlite3.Connection,
        job_id: int,
        level: str,
        message: str
    ) -> None:
        """Log a message for a job."""
        conn.execute(
            "INSERT INTO job_logs (job_id, timestamp, level, message) VALUES (?, ?, ?, ?)",
            (job_id, datetime.now().isoformat(), level, message)
        )

    def _record_memory(
        self,
        conn: sqlite3.Connection,
        job_id: int,
        gpu_id: int,
        memory_gb: float
    ) -> None:
        """Record a memory snapshot."""
        conn.execute(
            "INSERT INTO memory_snapshots (job_id, timestamp, gpu_id, memory_used_gb) VALUES (?, ?, ?, ?)",
            (job_id, datetime.now().isoformat(), gpu_id, memory_gb)
        )

    def get_job_logs(self, job_id: int) -> List[Dict[str, Any]]:
        """Get logs for a job."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM job_logs WHERE job_id = ? ORDER BY id",
                (job_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_memory_stats(self, job_id: int) -> Dict[str, float]:
        """Get memory statistics for a job."""
        with self._get_connection() as conn:
            row = conn.execute(
                """SELECT
                    MIN(memory_used_gb) as min_memory,
                    MAX(memory_used_gb) as max_memory,
                    AVG(memory_used_gb) as avg_memory
                FROM memory_snapshots WHERE job_id = ?""",
                (job_id,)
            ).fetchone()
            if row and row['max_memory'] is not None:
                return {
                    'min_gb': row['min_memory'],
                    'max_gb': row['max_memory'],
                    'avg_gb': row['avg_memory'],
                }
        return {'min_gb': 0, 'max_gb': 0, 'avg_gb': 0}

    def _build_train_command(self, job: TrainingJob) -> List[str]:
        """Build the training command for a job."""
        cmd = [
            sys.executable, '-m', 'rave', 'train',
            '--name', job.instrument_name,
            '--config', job.config,
            '--db_path', job.db_path,
            '--out_path', job.output_path,
            '--max_steps', str(job.max_steps),
            '--batch', str(job.batch_size),
        ]

        if job.gpu_id is not None:
            cmd.extend(['--gpu', str(job.gpu_id)])

        return cmd

    def _launch_job(self, job: TrainingJob) -> bool:
        """
        Launch a training job.

        Args:
            job: Job to launch

        Returns:
            True if launched successfully
        """
        # Determine GPU to use
        gpu_id = job.gpu_id
        if gpu_id is None:
            gpu_id = self.gpu_ids[0]

        # Check if we can launch
        if not self.gpu_monitor.can_launch_job(gpu_id, self.memory_per_job_gb):
            logger.info(f"Not enough memory on GPU {gpu_id} to launch job {job.id}")
            return False

        # Build command
        cmd = self._build_train_command(job)
        logger.info(f"Launching job {job.id}: {' '.join(cmd)}")

        try:
            # Set up environment
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            # Launch process
            log_file = os.path.join(job.output_path, f'{job.instrument_name}_train.log')
            os.makedirs(job.output_path, exist_ok=True)

            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(Path(__file__).parent.parent.parent)
                )

            # Update job status
            job.status = JobStatus.RUNNING
            job.pid = process.pid
            job.gpu_id = gpu_id
            job.started_at = datetime.now().isoformat()
            self.update_job(job)

            # Track running process
            self.running_jobs[job.id] = process

            with self._get_connection() as conn:
                self._log_job(conn, job.id, "INFO",
                              f"Job started on GPU {gpu_id}, PID {process.pid}")

            return True

        except Exception as e:
            logger.error(f"Failed to launch job {job.id}: {e}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            self.update_job(job)

            with self._get_connection() as conn:
                self._log_job(conn, job.id, "ERROR", f"Launch failed: {e}")

            return False

    def _check_running_jobs(self) -> None:
        """Check status of running jobs."""
        completed_ids = []

        for job_id, process in self.running_jobs.items():
            job = self.get_job(job_id)
            if job is None:
                completed_ids.append(job_id)
                continue

            # Record memory usage
            if job.gpu_id is not None:
                memory_gb = self.gpu_monitor.get_used_memory_gb(job.gpu_id)
                with self._get_connection() as conn:
                    self._record_memory(conn, job_id, job.gpu_id, memory_gb)

                # Update peak memory
                if memory_gb > job.peak_memory_gb:
                    job.peak_memory_gb = memory_gb
                    self.update_job(job)

            # Check if process completed
            retcode = process.poll()
            if retcode is not None:
                completed_ids.append(job_id)

                if retcode == 0:
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.now().isoformat()
                    self.update_job(job)

                    with self._get_connection() as conn:
                        self._log_job(conn, job_id, "INFO",
                                      f"Job completed successfully")

                    logger.info(f"Job {job_id} ({job.instrument_name}) completed")

                    if self.on_job_complete:
                        self.on_job_complete(job)

                else:
                    # Job failed
                    job.error_message = f"Process exited with code {retcode}"

                    # Check if we should retry
                    if job.retry_count < job.max_retries:
                        job.status = JobStatus.RETRYING
                        job.retry_count += 1
                        self.update_job(job)

                        with self._get_connection() as conn:
                            self._log_job(conn, job_id, "WARNING",
                                          f"Job failed, retrying ({job.retry_count}/{job.max_retries})")

                        logger.warning(f"Job {job_id} failed, will retry")

                        # Reset for retry
                        job.status = JobStatus.PENDING
                        job.pid = None
                        self.update_job(job)

                    else:
                        job.status = JobStatus.FAILED
                        job.completed_at = datetime.now().isoformat()
                        self.update_job(job)

                        with self._get_connection() as conn:
                            self._log_job(conn, job_id, "ERROR",
                                          f"Job failed after {job.max_retries} retries")

                        logger.error(f"Job {job_id} ({job.instrument_name}) failed")

                        if self.on_job_failed:
                            self.on_job_failed(job)

        # Remove completed from tracking
        for job_id in completed_ids:
            del self.running_jobs[job_id]

    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        logger.info("Scheduler started")

        while not self._stop_event.is_set():
            try:
                # Check running jobs
                self._check_running_jobs()

                # Get number of running jobs
                running_count = len(self.running_jobs)

                # Launch new jobs if capacity available
                if running_count < self.max_parallel_jobs:
                    pending = self.get_pending_jobs()
                    for job in pending:
                        if len(self.running_jobs) >= self.max_parallel_jobs:
                            break
                        self._launch_job(job)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")

            # Sleep before next check
            self._stop_event.wait(timeout=30)

        logger.info("Scheduler stopped")

    def start(self) -> None:
        """Start the scheduler."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            logger.warning("Scheduler already running")
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._scheduler_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        logger.info("Training queue scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Training queue scheduler stopped")

    def cancel_job(self, job_id: int) -> bool:
        """
        Cancel a job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled
        """
        job = self.get_job(job_id)
        if not job:
            return False

        if job.status == JobStatus.RUNNING:
            # Kill the process
            if job_id in self.running_jobs:
                process = self.running_jobs[job_id]
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                del self.running_jobs[job_id]

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now().isoformat()
        self.update_job(job)

        with self._get_connection() as conn:
            self._log_job(conn, job_id, "INFO", "Job cancelled")

        return True

    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status."""
        jobs = self.get_all_jobs()

        status_counts = {s.value: 0 for s in JobStatus}
        for job in jobs:
            status_counts[job.status.value] += 1

        # Get GPU memory status
        gpu_status = []
        for gpu_id in self.gpu_ids:
            total = self.gpu_monitor.get_total_memory_gb(gpu_id)
            used = self.gpu_monitor.get_used_memory_gb(gpu_id)
            gpu_status.append({
                'gpu_id': gpu_id,
                'total_gb': total,
                'used_gb': used,
                'available_gb': total - used,
            })

        return {
            'total_jobs': len(jobs),
            'status_counts': status_counts,
            'running_jobs': len(self.running_jobs),
            'max_parallel': self.max_parallel_jobs,
            'gpu_status': gpu_status,
        }


# Global queue instance
_queue: Optional[TrainingQueue] = None


def get_queue(db_path: Optional[str] = None, **kwargs) -> TrainingQueue:
    """
    Get or create the global queue instance.

    Args:
        db_path: Path to database file
        **kwargs: Additional arguments for TrainingQueue

    Returns:
        TrainingQueue instance
    """
    global _queue

    if db_path is None:
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data',
            'training_queue.db'
        )

    if _queue is None or _queue.db_path != db_path:
        _queue = TrainingQueue(db_path, **kwargs)

    return _queue
