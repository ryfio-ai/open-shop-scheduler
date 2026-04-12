from __future__ import annotations
import time
from typing import Any, Dict, List, Optional

TASKS: Dict[str, Dict[str, Any]] = {
    "easy_single_machine": {
        "id": "easy_single_machine",
        "description": "Schedule 5 jobs on 1 machine to minimise total tardiness.",
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
        "description": "Schedule 6 jobs on 2 machines with family changeover penalties.",
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
        "machines": [{"machine_id": 0, "name": "M1"}, {"machine_id": 1, "name": "M2"}],
        "changeover_penalty": 2,
    },
    "hard_dynamic_arrivals": {
        "id": "hard_dynamic_arrivals",
        "description": "Schedule 8 jobs on 3 machines with dynamic arrivals.",
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
        "machines": [{"machine_id": 0, "name": "M1"}, {"machine_id": 1, "name": "M2"}, {"machine_id": 2, "name": "M3"}],
        "changeover_penalty": 2,
    },
}

def grade_schedule(task: Dict[str, Any], assignments: List[Dict[str, Any]]) -> float:
    if not assignments:
        return 0.1
    jobs_by_id = {j["job_id"]: j for j in task["jobs"]}
    valid_machine_ids = {m["machine_id"] for m in task["machines"]}
    changeover = task.get("changeover_penalty", 0)
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
            penalty += 0.1; continue
        if jid not in jobs_by_id:
            penalty += 0.1; continue
        if jid in assigned_job_ids:
            penalty += 0.05; continue
        job = jobs_by_id[jid]
        setup = changeover if (machine_last_family[mid] and machine_last_family[mid] != job["family"]) else 0
        start = max(machine_time[mid], float(job["arrival_time"]))
        finish = start + setup + job["processing_time"]
        machine_time[mid] = finish
        machine_last_family[mid] = job["family"]
        assigned_job_ids.add(jid)
        valid_count += 1
        total_tardiness += max(0.0, finish - job["due_date"])
    coverage = valid_count / max(len(task["jobs"]), 1)
    max_tard = sum(j["processing_time"] + changeover for j in task["jobs"]) or 1.0
    tardiness_score = max(0.0, 1.0 - total_tardiness / max_tard)
    raw = 0.5 * coverage + 0.5 * tardiness_score - penalty
    return round(max(0.01, min(0.99, raw)), 4)

class ShopEnvironment:
    def __init__(self, task: Dict[str, Any]):
        self.task = task
        self.current_time = 0.0
        self.step_count = 0
        self.max_steps = task.get("max_steps", 20)
        self.assignments_log: List[Dict] = []
        self.machine_state = {
            m["machine_id"]: {"machine_id": m["machine_id"], "name": m["name"],
                              "status": "idle", "current_job": None,
                              "current_family": None, "available_at": 0.0}
            for m in task["machines"]
        }
        self.jobs = {j["job_id"]: dict(j, status="pending") for j in task["jobs"]}
        self.last_action_error: Optional[str] = None

    def get_observation(self) -> Dict[str, Any]:
        return {
            "task_id": self.task["id"],
            "current_time": self.current_time,
            "step": self.step_count,
            "machines": list(self.machine_state.values()),
            "jobs_pending": [j for j in self.jobs.values() if j["status"] == "pending" and j["arrival_time"] <= self.current_time],
            "jobs_in_progress": [j for j in self.jobs.values() if j["status"] == "in_progress"],
            "completed_jobs": [j for j in self.jobs.values() if j["status"] == "completed"],
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
            if machine is None: errors.append(f"Unknown machine {mid}"); continue
            if job is None: errors.append(f"Unknown job {jid}"); continue
            if job["status"] != "pending": errors.append(f"Job {jid} not pending"); continue
            if job["arrival_time"] > self.current_time: errors.append(f"Job {jid} not arrived"); continue
            changeover = self.task.get("changeover_penalty", 0) if (machine["current_family"] and machine["current_family"] != job["family"]) else 0
            start = max(machine["available_at"], float(job["arrival_time"]))
            finish = start + changeover + job["processing_time"]
            machine["status"] = "busy"; machine["current_job"] = jid; machine["current_family"] = job["family"]; machine["available_at"] = finish
            job["status"] = "completed"; job["completion_time"] = finish
            reward += max(0.0, 1.0 - max(0.0, finish - job["due_date"]) / max(job["due_date"], 1))
            self.assignments_log.append({"machine_id": mid, "job_id": jid, "start": start, "finish": finish})
        busy = [m for m in self.machine_state.values() if m["status"] == "busy"]
        if busy:
            self.current_time = min(m["available_at"] for m in busy)
            for m in busy:
                if m["available_at"] <= self.current_time: m["status"] = "idle"; m["current_job"] = None
        if errors: self.last_action_error = "; ".join(errors)
        done = self.is_done()
        return {"observation": self.get_observation(), "reward": round(reward, 4), "done": done,
                "score": grade_schedule(self.task, self.assignments_log) if done else None,
                "info": {"step": self.step_count, "errors": errors}}

    def is_done(self) -> bool:
        return all(j["status"] == "completed" for j in self.jobs.values()) or self.step_count >= self.max_steps
