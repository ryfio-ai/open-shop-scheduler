"""
server/environment.py
Core environment logic for OpenShop Scheduler.

Defines:
  TASKS         — dict of the 3 required tasks (easy / medium / hard)
  grade_schedule — deterministic grader, returns float in [0.0, 1.0]
  ShopEnvironment — stateful RL environment for a single episode
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

# ── Task definitions ──────────────────────────────────────────────────────────

TASKS: Dict[str, Dict[str, Any]] = {
    "easy_single_machine": {
        "id": "easy_single_machine",
        "description": (
            "Schedule 5 jobs on 1 machine to minimise total tardiness. "
            "Jobs have fixed processing times and deadlines."
        ),
        "difficulty": "easy",
        "num_jobs": 5,
        "num_machines": 1,
        "max_steps": 10,
        "jobs": [
            {"job_id": "J1", "family": "A", "processing_time": 3, "due_date": 5,  "arrival_time": 0},
            {"job_id": "J2", "family": "A", "processing_time": 2, "due_date": 4,  "arrival_time": 0},
            {"job_id": "J3", "family": "B", "processing_time": 5, "due_date": 10, "arrival_time": 0},
            {"job_id": "J4", "family": "B", "processing_time": 1, "due_date": 3,  "arrival_time": 0},
            {"job_id": "J5", "family": "A", "processing_time": 4, "due_date": 12, "arrival_time": 0},
        ],
        "machines": [{"machine_id": 0, "name": "M1"}],
        "changeover_penalty": 0,
    },
    "medium_parallel_changeover": {
        "id": "medium_parallel_changeover",
        "description": (
            "Schedule 6 jobs on 2 parallel machines. "
            "Switching job families on a machine incurs a 2-unit setup penalty."
        ),
        "difficulty": "medium",
        "num_jobs": 6,
        "num_machines": 2,
        "max_steps": 15,
        "jobs": [
            {"job_id": "J1", "family": "A", "processing_time": 4, "due_date": 8,  "arrival_time": 0},
            {"job_id": "J2", "family": "B", "processing_time": 3, "due_date": 6,  "arrival_time": 0},
            {"job_id": "J3", "family": "A", "processing_time": 2, "due_date": 7,  "arrival_time": 0},
            {"job_id": "J4", "family": "B", "processing_time": 5, "due_date": 12, "arrival_time": 0},
            {"job_id": "J5", "family": "A", "processing_time": 3, "due_date": 10, "arrival_time": 0},
            {"job_id": "J6", "family": "B", "processing_time": 4, "due_date": 14, "arrival_time": 0},
        ],
        "machines": [
            {"machine_id": 0, "name": "M1"},
            {"machine_id": 1, "name": "M2"},
        ],
        "changeover_penalty": 2,
    },
    "hard_dynamic_arrivals": {
        "id": "hard_dynamic_arrivals",
        "description": (
            "Schedule 8 jobs on 3 machines with dynamic arrivals. "
            "Jobs arrive at different times requiring real-time adaptation."
        ),
        "difficulty": "hard",
        "num_jobs": 8,
        "num_machines": 3,
        "max_steps": 30,
        "jobs": [
            {"job_id": "J1", "family": "A", "processing_time": 5, "due_date": 10, "arrival_time": 0},
            {"job_id": "J2", "family": "B", "processing_time": 3, "due_date": 8,  "arrival_time": 0},
            {"job_id": "J3", "family": "C", "processing_time": 4, "due_date": 12, "arrival_time": 5},
            {"job_id": "J4", "family": "A", "processing_time": 6, "due_date": 18, "arrival_time": 5},
            {"job_id": "J5", "family": "B", "processing_time": 2, "due_date": 15, "arrival_time": 10},
            {"job_id": "J6", "family": "C", "processing_time": 7, "due_date": 22, "arrival_time": 10},
            {"job_id": "J7", "family": "A", "processing_time": 3, "due_date": 25, "arrival_time": 20},
            {"job_id": "J8", "family": "B", "processing_time": 5, "due_date": 30, "arrival_time": 20},
        ],
        "machines": [
            {"machine_id": 0, "name": "M1"},
            {"machine_id": 1, "name": "M2"},
            {"machine_id": 2, "name": "M3"},
        ],
        "changeover_penalty": 2,
    },
}


# ── Grader ────────────────────────────────────────────────────────────────────

def grade_schedule(
    task: Dict[str, Any],
    assignments: List[Dict[str, Any]],
) -> float:
    """
    Deterministic grader. Returns a score in [0.0, 1.0].

    Scoring:
      - Start with 1.0
      - Deduct for each job that is tardy (completion_time > due_date)
      - Deduct for invalid assignments (unknown job_id / machine_id)
      - Bonus for assigning all jobs

    If no assignments are provided we return a baseline of 0.1
    (the environment ran but agent did nothing).
    """
    if not assignments:
        return 0.1

    jobs_by_id = {j["job_id"]: j for j in task["jobs"]}
    valid_machine_ids = {m["machine_id"] for m in task["machines"]}
    changeover = task.get("changeover_penalty", 0)

    # Simulate simple schedule: track completion time per machine
    machine_time: Dict[int, float] = {m["machine_id"]: 0.0 for m in task["machines"]}
    machine_last_family: Dict[int, Optional[str]] = {m["machine_id"]: None for m in task["machines"]}

    assigned_job_ids = set()
    total_tardiness = 0.0
    penalty = 0.0
    valid_count = 0

    for asgn in assignments:
        mid = asgn.get("machine_id")
        jid = asgn.get("job_id")

        if mid not in valid_machine_ids:
            penalty += 0.1
            continue
        if jid not in jobs_by_id:
            penalty += 0.1
            continue
        if jid in assigned_job_ids:
            penalty += 0.05  # duplicate assignment
            continue

        job = jobs_by_id[jid]
        setup = changeover if (machine_last_family[mid] and machine_last_family[mid] != job["family"]) else 0
        start = max(machine_time[mid], float(job["arrival_time"]))
        finish = start + setup + job["processing_time"]

        machine_time[mid] = finish
        machine_last_family[mid] = job["family"]
        assigned_job_ids.add(jid)
        valid_count += 1

        tardiness = max(0.0, finish - job["due_date"])
        total_tardiness += tardiness

    # Fraction of jobs assigned
    coverage = valid_count / max(len(task["jobs"]), 1)

    # Normalise tardiness penalty (max possible tardiness used as denominator)
    max_possible_tardiness = sum(
        j["processing_time"] + changeover for j in task["jobs"]
    ) or 1.0
    tardiness_score = max(0.0, 1.0 - total_tardiness / max_possible_tardiness)

    raw_score = 0.5 * coverage + 0.5 * tardiness_score - penalty
    return round(max(0.0, min(1.0, raw_score)), 4)


# ── Environment ───────────────────────────────────────────────────────────────

class ShopEnvironment:
    """Stateful RL environment for one episode."""

    def __init__(self, task: Dict[str, Any]):
        self.task = task
        self.current_time = 0.0
        self.step_count = 0
        self.max_steps = task.get("max_steps", 20)
        self.assignments_log: List[Dict] = []

        # Machine state
        self.machine_state = {
            m["machine_id"]: {
                "machine_id": m["machine_id"],
                "name": m["name"],
                "status": "idle",
                "current_job": None,
                "current_family": None,
                "available_at": 0.0,
            }
            for m in task["machines"]
        }

        # Job state
        self.jobs = {j["job_id"]: dict(j, status="pending") for j in task["jobs"]}
        self.last_action_error: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_observation(self) -> Dict[str, Any]:
        pending = [
            j for j in self.jobs.values()
            if j["status"] == "pending" and j["arrival_time"] <= self.current_time
        ]
        in_progress = [j for j in self.jobs.values() if j["status"] == "in_progress"]
        completed = [j for j in self.jobs.values() if j["status"] == "completed"]

        return {
            "task_id": self.task["id"],
            "current_time": self.current_time,
            "step": self.step_count,
            "machines": list(self.machine_state.values()),
            "jobs_pending": pending,
            "jobs_in_progress": in_progress,
            "completed_jobs": completed,
            "last_action_error": self.last_action_error,
        }

    def step(self, assignments: List[Dict], reasoning: str = "") -> Dict[str, Any]:
        self.step_count += 1
        self.last_action_error = None
        errors = []
        reward = 0.0

        for asgn in assignments:
            mid = asgn.get("machine_id")
            jid = asgn.get("job_id")

            machine = self.machine_state.get(mid)
            job = self.jobs.get(jid)

            if machine is None:
                errors.append(f"Unknown machine_id: {mid}")
                continue
            if job is None:
                errors.append(f"Unknown job_id: {jid}")
                continue
            if job["status"] != "pending":
                errors.append(f"Job {jid} is not pending (status={job['status']})")
                continue
            if job["arrival_time"] > self.current_time:
                errors.append(f"Job {jid} has not arrived yet (arrives at t={job['arrival_time']})")
                continue
            if machine["status"] == "busy":
                errors.append(f"Machine {mid} is busy")
                continue

            # Apply changeover
            changeover = 0
            if (
                machine["current_family"]
                and machine["current_family"] != job["family"]
                and self.task.get("changeover_penalty", 0) > 0
            ):
                changeover = self.task["changeover_penalty"]

            start = max(machine["available_at"], float(job["arrival_time"]))
            finish = start + changeover + job["processing_time"]

            # Update machine
            machine["status"] = "busy"
            machine["current_job"] = jid
            machine["current_family"] = job["family"]
            machine["available_at"] = finish

            # Update job
            job["status"] = "completed"
            job["completion_time"] = finish

            # Reward shaping
            tardiness = max(0.0, finish - job["due_date"])
            reward += max(0.0, 1.0 - tardiness / max(job["due_date"], 1))

            self.assignments_log.append(
                {"machine_id": mid, "job_id": jid, "start": start, "finish": finish}
            )

        # Advance time to next machine availability
        busy_machines = [m for m in self.machine_state.values() if m["status"] == "busy"]
        if busy_machines:
            next_free = min(m["available_at"] for m in busy_machines)
            if next_free > self.current_time:
                self.current_time = next_free
                for m in busy_machines:
                    if m["available_at"] <= self.current_time:
                        m["status"] = "idle"
                        m["current_job"] = None

        if errors:
            self.last_action_error = "; ".join(errors)

        done = self.is_done()
        final_score = None
        if done:
            final_score = grade_schedule(self.task, self.assignments_log)

        return {
            "observation": self.get_observation(),
            "reward": round(reward, 4),
            "done": done,
            "score": final_score,
            "info": {
                "step": self.step_count,
                "errors": errors,
                "reasoning": reasoning,
            },
        }

    def is_done(self) -> bool:
        all_completed = all(j["status"] == "completed" for j in self.jobs.values())
        timed_out = self.step_count >= self.max_steps
        return all_completed or timed_out
