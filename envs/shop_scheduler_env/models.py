from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field

MachineStatus = Literal["idle", "processing", "setup"]
JobStatus = Literal["pending", "processing", "completed"]

class MachineSnapshot(BaseModel):
    machine_id: str
    status: MachineStatus
    current_job_id: Optional[str] = None
    family: Optional[str] = None
    time_remaining: int = 0

class JobSnapshot(BaseModel):
    job_id: str
    family: str
    processing_time: int
    remaining_time: int
    due_time: int
    priority: Literal["low", "medium", "high", "rush"] = "medium"
    status: JobStatus = "pending"
    arrival_time: int = 0
    assigned_machine_id: Optional[str] = None
    completion_time: Optional[int] = None

class Observation(BaseModel):
    task_id: str
    current_time: int
    max_time: int
    step_count: int
    max_steps: int
    machines: List[MachineSnapshot]
    jobs_pending: List[JobSnapshot]
    jobs_in_progress: List[JobSnapshot]
    completed_jobs: List[JobSnapshot]
    valid_actions_hint: List[str] = Field(default_factory=list)
    last_action_error: Optional[str] = None

class MachineAssignment(BaseModel):
    machine_id: str
    job_id: Optional[str] = None

class Action(BaseModel):
    assignments: List[MachineAssignment]
    reasoning: Optional[str] = None

class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    on_time_bonus: float = 0.0
    tardiness_penalty: float = 0.0
    idle_penalty: float = 0.0
    invalid_action_penalty: float = 0.0
    info: Dict[str, float] = Field(default_factory=dict)

class EnvState(BaseModel):
    task_id: str
    scenario_name: str
    current_time: int
    max_time: int
    step_count: int
    max_steps: int
    done: bool
    total_raw_reward: float
    normalized_score: float
    machines: List[MachineSnapshot]
    jobs: List[JobSnapshot]
    action_history: List[Action] = Field(default_factory=list)
    reward_history: List[Reward] = Field(default_factory=list)
    last_action_error: Optional[str] = None
