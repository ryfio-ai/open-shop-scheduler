import os, time, uuid, sys
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── path injection so we can import from 'server' ────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import TASKS, ShopEnvironment, grade_schedule

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
    assignments: List[Assignment] = []

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/tasks")
def list_tasks():
    # Return as flat list for max compatibility with Scaler validator
    return [
        {
            "id": t["id"],
            "description": t["description"],
            "difficulty": t["difficulty"],
            "grader": True,
            "has_grader": True,
            "num_jobs": t.get("num_jobs", 0),
            "num_machines": t.get("num_machines", 0),
        }
        for t in TASKS.values()
    ]

@app.post("/grader")
def grader(req: Optional[GraderRequest] = None):
    # Support empty body for robust validation
    task_id = req.task_id if req else "easy_single_machine"
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    
    assignments_raw = [{"machine_id": a.machine_id, "job_id": a.job_id} for a in req.assignments] if req else []
    score = grade_schedule(task=task, assignments=assignments_raw)
    return {"task_id": task_id, "score": score, "reward": score, "grader": "deterministic"}

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
