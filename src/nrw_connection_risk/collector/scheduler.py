"""Minimal interval scheduler: runs the job that is due next, never in parallel."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Job:
    name: str
    interval_s: float
    action: Callable[[], None]
    next_run: float = 0.0
    runs: int = field(default=0)


class Scheduler:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.jobs: list[Job] = []

    def add(self, name: str, interval_s: float, action: Callable[[], None], offset_s: float = 0.0) -> Job:
        job = Job(name=name, interval_s=interval_s, action=action, next_run=self.clock() + offset_s)
        self.jobs.append(job)
        return job

    def next_job(self) -> Job:
        return min(self.jobs, key=lambda j: j.next_run)

    def run_job(self, job: Job) -> None:
        job.action()
        job.runs += 1
        job.next_run += job.interval_s
        now = self.clock()
        if job.next_run < now:            # fell behind: skip missed slots instead of bursting
            job.next_run = now
