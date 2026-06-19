#!/usr/bin/env python
"""Persistent experiment queue with SQLite backend for resumable execution."""
import sqlite3
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import threading


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentJob:
    id: str
    variant: str
    benchmark: str
    config: Dict[str, Any]
    priority: float
    status: str = JobStatus.PENDING.value
    created_at: float = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    checkpoint_path: Optional[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}


class PersistentQueue:
    """Thread-safe persistent queue with priority ordering."""
    
    def __init__(self, db_path: str = "experiment_queue.db"):
        self.db_path = Path(db_path)
        self.lock = threading.RLock()
        self._init_db()
    
    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    variant TEXT NOT NULL,
                    benchmark TEXT NOT NULL,
                    config TEXT NOT NULL,
                    priority REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    result TEXT,
                    error TEXT,
                    checkpoint_path TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status_priority 
                ON jobs(status, priority DESC, created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_variant_benchmark 
                ON jobs(variant, benchmark)
            """)
    
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def submit(self, job: ExperimentJob) -> str:
        """Add job to queue."""
        with self.lock, self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO jobs 
                (id, variant, benchmark, config, priority, status, created_at, 
                 started_at, completed_at, result, error, checkpoint_path, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.id, job.variant, job.benchmark, json.dumps(job.config),
                job.priority, job.status, job.created_at,
                job.started_at, job.completed_at,
                json.dumps(job.result) if job.result else None,
                job.error, job.checkpoint_path,
                json.dumps(job.metadata) if job.metadata else None
            ))
        return job.id
    
    def submit_batch(self, jobs: List[ExperimentJob]) -> List[str]:
        """Submit multiple jobs atomically."""
        ids = []
        with self.lock, self._conn() as conn:
            for job in jobs:
                conn.execute("""
                    INSERT OR REPLACE INTO jobs 
                    (id, variant, benchmark, config, priority, status, created_at,
                     started_at, completed_at, result, error, checkpoint_path, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job.id, job.variant, job.benchmark, json.dumps(job.config),
                    job.priority, job.status, job.created_at,
                    job.started_at, job.completed_at,
                    json.dumps(job.result) if job.result else None,
                    job.error, job.checkpoint_path,
                    json.dumps(job.metadata) if job.metadata else None
                ))
                ids.append(job.id)
        return ids
    
    def pop_next(self, worker_id: str = "") -> Optional[ExperimentJob]:
        """Atomically claim highest-priority pending job."""
        with self.lock, self._conn() as conn:
            row = conn.execute("""
                SELECT * FROM jobs 
                WHERE status = ? 
                ORDER BY priority DESC, created_at 
                LIMIT 1
            """, (JobStatus.PENDING.value,)).fetchone()
            
            if not row:
                return None
            
            job_id = row["id"]
            now = time.time()
            conn.execute("""
                UPDATE jobs SET status = ?, started_at = ?, metadata = 
                json_set(COALESCE(metadata, '{}'), '$.worker_id', ?)
                WHERE id = ?
            """, (JobStatus.RUNNING.value, now, worker_id, job_id))
            
            return self._row_to_job(row)
    
    def complete(self, job_id: str, result: Dict, checkpoint_path: str = None):
        """Mark job completed with result."""
        with self.lock, self._conn() as conn:
            conn.execute("""
                UPDATE jobs SET status = ?, completed_at = ?, result = ?, 
                checkpoint_path = ? WHERE id = ?
            """, (JobStatus.COMPLETED.value, time.time(), json.dumps(result), 
                  checkpoint_path, job_id))
    
    def fail(self, job_id: str, error: str):
        """Mark job failed."""
        with self.lock, self._conn() as conn:
            conn.execute("""
                UPDATE jobs SET status = ?, completed_at = ?, error = ? WHERE id = ?
            """, (JobStatus.FAILED.value, time.time(), error, job_id))
    
    def update_checkpoint(self, job_id: str, checkpoint_path: str):
        """Update checkpoint path for running job."""
        with self.lock, self._conn() as conn:
            conn.execute("""
                UPDATE jobs SET checkpoint_path = ? WHERE id = ?
            """, (checkpoint_path, job_id))
    
    def get_job(self, job_id: str) -> Optional[ExperimentJob]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return self._row_to_job(row) if row else None
    
    def get_jobs(self, status: JobStatus = None, 
                 variant: str = None, benchmark: str = None,
                 limit: int = 100) -> List[ExperimentJob]:
        """Query jobs with filters."""
        query = "SELECT * FROM jobs WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status.value)
        if variant:
            query += " AND variant = ?"
            params.append(variant)
        if benchmark:
            query += " AND benchmark = ?"
            params.append(benchmark)
        query += " ORDER BY priority DESC, created_at LIMIT ?"
        params.append(limit)
        
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_job(r) for r in rows]
    
    def get_pending_count(self) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ?", 
                (JobStatus.PENDING.value,)
            ).fetchone()[0]
    
    def get_stats(self) -> Dict:
        with self._conn() as conn:
            stats = {}
            for status in JobStatus:
                count = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status = ?", (status.value,)
                ).fetchone()[0]
                stats[status.value] = count
            return stats
    
    def reset_stale_jobs(self, max_age_hours: float = 24):
        """Reset RUNNING jobs older than max_age (worker likely died)."""
        cutoff = time.time() - max_age_hours * 3600
        with self.lock, self._conn() as conn:
            conn.execute("""
                UPDATE jobs SET status = ?, started_at = NULL, error = ?
                WHERE status = ? AND started_at < ?
            """, (JobStatus.PENDING.value, "Worker timeout - requeued",
                  JobStatus.RUNNING.value, cutoff))
    
    def _row_to_job(self, row) -> ExperimentJob:
        return ExperimentJob(
            id=row["id"],
            variant=row["variant"],
            benchmark=row["benchmark"],
            config=json.loads(row["config"]),
            priority=row["priority"],
            status=row["status"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
            checkpoint_path=row["checkpoint_path"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )


def create_job(variant: str, benchmark: str, config: Dict, 
               priority: float = 1.0) -> ExperimentJob:
    """Factory for creating experiment jobs."""
    return ExperimentJob(
        id=str(uuid.uuid4())[:8],
        variant=variant,
        benchmark=benchmark,
        config=config,
        priority=priority,
    )


if __name__ == "__main__":
    # Demo
    q = PersistentQueue("test_queue.db")
    
    # Submit some jobs
    for i in range(5):
        job = create_job("baseline", "split_mnist", {"lr": 1e-3, "seed": 42+i}, priority=5-i)
        q.submit(job)
    
    print("Stats:", q.get_stats())
    print("Pending:", q.get_pending_count())
    
    # Pop and complete
    job = q.pop_next("worker1")
    print(f"Popped: {job.id}")
    q.complete(job.id, {"accuracy": 0.95})
    
    print("Stats:", q.get_stats())
