import os, time, uuid, sys
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── path injection so we can import from 'server' ────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import TASKS, ShopEnvironment
from envs.shop_scheduler_env.graders import (
    grade_easy_single_machine,
    grade_medium_parallel_changeover,
    grade_hard_dynamic_arrivals
)

app = FastAPI(title="OpenShop Scheduler", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_sessions: Dict[str, ShopEnvironment] = {}

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
    episode_state: dict

class GraderResponse(BaseModel):
    score: float
    feedback: str

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/tasks")
def list_tasks():
    """Return available tasks - REQUIRED for validation"""
    return {
        "tasks": [
            {
                "id": t["id"],
                "task_id": t["id"],
                "name": t.get("name", t["id"].replace("_", " ").title()),
                "description": t["description"],
                "difficulty": t["difficulty"],
            }
            for t in TASKS.values()
        ]
    }

@app.post("/grader")
def grader_endpoint(request: GraderRequest):
    """Grade an episode - REQUIRED for Phase 2 validation"""
    task_id = request.task_id
    
    # Route to appropriate grader based on task
    try:
        if task_id == "easy_single_machine":
            score = grade_easy_single_machine(request.episode_state)
        elif task_id == "medium_parallel_changeover":
            score = grade_medium_parallel_changeover(request.episode_state)
        elif task_id == "hard_dynamic_arrivals":
            score = grade_hard_dynamic_arrivals(request.episode_state)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown task: {task_id}")
    except Exception as e:
        # Fallback to a safe low score if grading fails
        print(f"Grading error: {e}")
        score = 0.01

    # CRITICAL: Clamp score to (0.01, 0.99)
    score = float(max(0.01, min(0.99, score)))
    
    return GraderResponse(score=score, feedback="Graded successfully")

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    # Support empty body for robust validation
    task_id = req.task_id if req else "easy_single_machine"
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    
    episode_id = str(uuid.uuid4())
    env = ShopEnvironment(task)
    _sessions[episode_id] = env
    return {
        "episode_id": episode_id, 
        "task_id": task_id,
        "observation": env.get_observation(), 
        "reward": 0.0, 
        "done": False, 
        "info": {}
    }

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

@app.get("/state/{episode_id}")
def state(episode_id: str):
    env = _sessions.get(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Episode not found.")
    return {"episode_id": episode_id, "observation": env.get_observation(), "done": env.is_done()}

@app.get("/")
def root():
    return {"env": "open-shop-scheduler", "status": "ok", "endpoints": ["/health", "/tasks", "/grader", "/reset", "/step", "/state"]}

def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
