"""In-memory job queue: map (parallel sub-agents) → reduce (single merge)."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal


def _sub_agent_prompt(challenge: str, index: int, total: int) -> str:
    c = challenge.strip()
    return (
        f"You are sub-agent {index + 1} of {total} running in parallel (like IDE sub-agents). "
        f"Give a concise partial answer, angle, or checklist item for the shared challenge. "
        f"Do not repeat others' roles; add distinct value.\n\n"
        f"Challenge:\n{c}"
    )


def _reduce_prompt(challenge: str, parts: list[dict[str, Any]]) -> str:
    lines = [
        "You synthesize parallel sub-agent outputs into one coherent final answer.",
        "Resolve disagreements briefly; prefer correctness and clarity.",
        "",
        f"Original challenge:\n{challenge.strip()}",
        "",
        "Sub-agent outputs:",
    ]
    for i, p in enumerate(parts, start=1):
        wid = p.get("worker_id") or "?"
        txt = (p.get("text") or "").strip() or "(no output)"
        lines.append(f"--- Agent {i} (worker {wid}) ---\n{txt}")
    lines.extend(["", "Integrated answer:"])
    return "\n".join(lines)


TaskKind = Literal["sub", "reduce"]
TaskStatus = Literal["pending", "assigned", "done"]
JobStatus = Literal["running", "done", "failed"]


@dataclass
class Task:
    id: str
    job_id: str
    kind: TaskKind
    status: TaskStatus
    prompt: str
    tools: list[str]
    force_cloud_fallback: bool = False
    assignee: str | None = None
    text: str = ""
    inference_source: str = ""
    error: str | None = None


@dataclass
class Job:
    id: str
    challenge: str
    parallel: int
    tools: list[str]
    force_cloud_fallback: bool = False
    nodes_available_at_create: int = 0
    status: JobStatus = "running"
    sub_task_ids: list[str] = field(default_factory=list)
    reduce_task_id: str | None = None
    final_answer: str | None = None
    error: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.jobs: dict[str, Job] = {}
        self.tasks: dict[str, Task] = {}
        self._pending: deque[str] = deque()
        self._workers: dict[str, float] = {}

    async def touch_worker(self, worker_id: str) -> None:
        """Record that a worker was seen (used by the swarm listener)."""
        async with self._lock:
            self._workers[worker_id] = time.monotonic()

    async def create_job(
        self,
        challenge: str,
        parallel: int | None,
        tools: list[str],
        *,
        force_cloud_fallback: bool = False,
        nodes_available: int | None = None,
    ) -> str:
        async with self._lock:
            nw = int(nodes_available) if nodes_available is not None else len(self._workers)
            if parallel is None:
                requested = min(32, max(1, nw)) if nw else 3
            else:
                rp = max(1, min(32, parallel))
                requested = rp
            effective = min(requested, nw) if nw else requested
            job_id = str(uuid.uuid4())
            job = Job(
                id=job_id,
                challenge=challenge.strip(),
                parallel=effective,
                tools=list(tools),
                force_cloud_fallback=force_cloud_fallback,
                nodes_available_at_create=nw,
            )
            self.jobs[job_id] = job
            for i in range(effective):
                tid = str(uuid.uuid4())
                task = Task(
                    id=tid,
                    job_id=job_id,
                    kind="sub",
                    status="pending",
                    prompt=_sub_agent_prompt(challenge, i, effective),
                    tools=list(tools),
                    force_cloud_fallback=force_cloud_fallback,
                )
                self.tasks[tid] = task
                job.sub_task_ids.append(tid)
                self._pending.append(tid)
            return job_id

    async def pop_pending(self) -> Task | None:
        """Pop the next pending task, leaving it unassigned."""
        async with self._lock:
            while self._pending:
                tid = self._pending.popleft()
                task = self.tasks.get(tid)
                if not task or task.status != "pending":
                    continue
                return task
            return None

    async def assign(self, task_id: str, worker_id: str) -> None:
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task or task.status != "pending":
                return
            task.status = "assigned"
            task.assignee = worker_id
            self._workers[worker_id] = time.monotonic()

    async def complete_task(
        self,
        task_id: str,
        worker_id: str,
        *,
        text: str,
        inference_source: str,
        error: str | None,
    ) -> bool:
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task or task.assignee != worker_id:
                return False
            if task.status == "done":
                return True
            task.status = "done"
            task.text = (text or "").strip()
            task.inference_source = inference_source or "local"
            task.error = error

            job = self.jobs.get(task.job_id)
            if not job:
                return True

            if task.kind == "sub":
                if all(self.tasks[tid].status == "done" for tid in job.sub_task_ids):
                    self._enqueue_reduce_locked(job)
            elif task.kind == "reduce":
                if error:
                    job.status = "failed"
                    job.error = error
                    job.final_answer = task.text or None
                else:
                    job.status = "done"
                    job.final_answer = task.text or ""
            return True

    def _enqueue_reduce_locked(self, job: Job) -> None:
        if job.reduce_task_id:
            return
        parts: list[dict[str, Any]] = []
        for tid in job.sub_task_ids:
            t = self.tasks[tid]
            parts.append(
                {
                    "task_id": tid,
                    "worker_id": t.assignee or "",
                    "text": t.text,
                    "inference_source": t.inference_source,
                    "error": t.error,
                }
            )
        rid = str(uuid.uuid4())
        rtask = Task(
            id=rid,
            job_id=job.id,
            kind="reduce",
            status="pending",
            prompt=_reduce_prompt(job.challenge, parts),
            tools=[],
            force_cloud_fallback=job.force_cloud_fallback,
        )
        self.tasks[rid] = rtask
        job.reduce_task_id = rid
        self._pending.append(rid)

    async def get_job_public(self, job_id: str) -> dict[str, Any] | None:
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            sub_results: list[dict[str, Any]] = []
            for tid in job.sub_task_ids:
                t = self.tasks.get(tid)
                if not t:
                    continue
                sub_results.append(
                    {
                        "task_id": tid,
                        "worker_id": t.assignee,
                        "status": t.status,
                        "text": t.text,
                        "inference_source": t.inference_source,
                        "error": t.error,
                    }
                )
            return {
                "job_id": job.id,
                "status": job.status,
                "challenge": job.challenge,
                "parallel": job.parallel,
                "nodes_available_at_create": job.nodes_available_at_create,
                "tools": job.tools,
                "force_cloud_fallback": job.force_cloud_fallback,
                "sub_results": sub_results,
                "reduce_task_id": job.reduce_task_id,
                "final_answer": job.final_answer,
                "error": job.error,
            }

    async def worker_stats(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "workers_seen": len(self._workers),
                "pending_tasks": sum(1 for t in self.tasks.values() if t.status == "pending"),
                "assigned_tasks": sum(1 for t in self.tasks.values() if t.status == "assigned"),
            }

