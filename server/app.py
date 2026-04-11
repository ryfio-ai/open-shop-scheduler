"""
OpenShop Scheduler — FastAPI server
Fixes Phase 2 Task Validation by exposing:
  GET  /tasks        → lists all 3 tasks with graders
  POST /grader       → grades a schedule for a given task_id
"""

import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

# ── path injection so we can import from 'server' ────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Import environment logic ─────────────────────────────────────────────────
from server.environment import (
    TASKS,
    ShopEnvironment,
    grade_schedule,
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="OpenShop Scheduler",
    description="Open-shop scheduling RL environment for the Meta PyTorch OpenEnv Hackathon",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ───────────────────────────────────────────────────
_sessions: Dict[str, ShopEnvironment] = {}


# ── Request / Response models ─────────────────────────────────────────────────
class Assignment(BaseModel):
    machine_id: int
    job_id: str


class Action(BaseModel):
    assignments: List[Assignment]
    reasoning: Optional[str] = ""


class ResetRequest(BaseModel):
    task_id: str = "easy_single_machine"


class GraderRequest(BaseModel):
    task_id: str
    assignments: List[Assignment]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


# ── /tasks  (REQUIRED by Phase-2 evaluator) ──────────────────────────────────
@app.get("/tasks")
def list_tasks():
    """Return all tasks that have an associated grader.
    The evaluator calls this and counts tasks where has_grader==True.
    Must return >= 3 tasks to pass Task Validation."""
    return {
        "tasks": [
            {
                "id": t["id"],
                "description": t["description"],
                "difficulty": t["difficulty"],
                "has_grader": True,          # ← critical field
                "num_jobs": t["num_jobs"],
                "num_machines": t["num_machines"],
            }
            for t in TASKS.values()
        ]
    }


# ── /grader  (REQUIRED by Phase-2 evaluator) ─────────────────────────────────
@app.post("/grader")
def grader(req: Optional[GraderRequest] = None):
    """Grade a schedule without running a full episode.
    Returns a score in [0.0, 1.0].
    The evaluator calls this per task to verify the grader works."""
    task_id = req.task_id if req else "easy_single_machine"
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    score = grade_schedule(
        task=task,
        assignments=[{"machine_id": a.machine_id, "job_id": a.job_id} for a in req.assignments] if req else [],
    )
    return {
        "task_id": task_id,
        "score": score,
        "reward": score,
        "grader": "deterministic",
    }


# ── /reset ────────────────────────────────────────────────────────────────────
@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    try:
        task_id = req.task_id if req else "easy_single_machine"
        task = TASKS.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

        episode_id = str(uuid.uuid4())
        env = ShopEnvironment(task)
        _sessions[episode_id] = env

        obs = env.get_observation()
        return {
            "episode_id": episode_id,
            "task_id": task_id,
            "observation": obs,
            "reward": 0.0,
            "done": False,
            "info": {},
        }
    except Exception as e:
        import traceback
        error_msg = f"Reset failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


# ── /step ─────────────────────────────────────────────────────────────────────
@app.post("/step/{episode_id}")
def step(episode_id: str, action: Action):
    env = _sessions.get(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Episode not found. Call /reset first.")

    result = env.step(
        assignments=[{"machine_id": a.machine_id, "job_id": a.job_id} for a in action.assignments],
        reasoning=action.reasoning,
    )
    if result["done"]:
        _sessions.pop(episode_id, None)

    return result


# ── /state ────────────────────────────────────────────────────────────────────
@app.get("/state/{episode_id}")
def state(episode_id: str):
    env = _sessions.get(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Episode not found.")
    return {
        "episode_id": episode_id,
        "observation": env.get_observation(),
        "done": env.is_done(),
    }

@app.get("/")
def root():
    return {
        "env": "open-shop-scheduler",
        "status": "ok",
        "endpoints": ["/health", "/tasks", "/grader", "/reset", "/step", "/state"]
    }

def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
