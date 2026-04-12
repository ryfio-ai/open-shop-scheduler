import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

from .models import (
    Observation, Action, Reward, EnvState,
    JobSnapshot, MachineSnapshot, MachineStatus, JobStatus
)
from .graders import grade_episode
from .rewards import compute_step_reward
from .tasks import load_scenario

class ShopSchedulerEnv:
    CHANGEOVER_TIME = 2

    def __init__(self, task_id: str = "easy_single_machine"):
        self.task_id = task_id
        self._state = None

    def reset(self) -> Observation:
        scenario = load_scenario(self.task_id)

        machines = [
            MachineSnapshot(**m) for m in scenario["machines"]
        ]
        jobs = [
            JobSnapshot(
                job_id=j["job_id"],
                family=j["family"],
                processing_time=j["processing_time"],
                remaining_time=j["processing_time"],
                due_time=j["due_time"],
                priority=j.get("priority", "medium"),
                arrival_time=j.get("arrival_time", 0),
            )
            for j in scenario["jobs"]
        ]

        self._state = EnvState(
            task_id=self.task_id,
            scenario_name=scenario["scenario_name"],
            current_time=0,
            max_time=scenario["max_time"],
            step_count=0,
            max_steps=scenario["max_steps"],
            done=False,
            total_raw_reward=0.0,
            normalized_score=0.001,
            machines=machines,
            jobs=jobs,
            action_history=[],
            reward_history=[],
            last_action_error=None,
        )
        return self._build_observation()

    def _build_observation(self) -> Observation:
        jobs_pending = [j for j in self._state.jobs if j.status == "pending" and self._state.current_time >= j.arrival_time]
        jobs_in_progress = [j for j in self._state.jobs if j.status == "processing"]
        completed_jobs = [j for j in self._state.jobs if j.status == "completed"]

        # Hint: only idle machines can be assigned
        idle_machine_ids = [m.machine_id for m in self._state.machines if m.status == "idle"]
        
        valid_actions = []
        if idle_machine_ids:
            valid_actions = [j.job_id for j in jobs_pending]

        return Observation(
            task_id=self._state.task_id,
            current_time=self._state.current_time,
            max_time=self._state.max_time,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            machines=self._state.machines,
            jobs_pending=jobs_pending,
            jobs_in_progress=jobs_in_progress,
            completed_jobs=completed_jobs,
            valid_actions_hint=valid_actions,
            last_action_error=self._state.last_action_error,
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._state.done:
            reward = Reward(value=0.0)
            return self._build_observation(), reward, True, {"error": "episode_already_done"}

        self._state.step_count += 1
        self._state.last_action_error = None
        
        # 1. Process assignments
        if action.assignments:
            for assignment in action.assignments:
                self._apply_assignment(assignment)

        # 2. Advance time
        # We advance until the next 'event' or by at least 1 unit if possible.
        # This prevents infinite loops of idle steps.
        self._advance_time()

        # 3. Compute reward
        reward = compute_step_reward(self._state, action)
        self._state.total_raw_reward += reward.value
        self._state.action_history.append(action)
        self._state.reward_history.append(reward)

        # 4. Update score
        self._state.normalized_score = grade_episode(self._state)

        # 5. Check done
        all_completed = all(j.status == "completed" for j in self._state.jobs)
        done = (
            self._state.step_count >= self._state.max_steps
            or self._state.current_time >= self._state.max_time
            or all_completed
        )
        self._state.done = done

        info = {
            "score": self._state.normalized_score,
            "last_action_error": self._state.last_action_error,
        }
        return self._build_observation(), reward, done, info

    def _apply_assignment(self, assignment) -> None:
        machine = next((m for m in self._state.machines if m.machine_id == assignment.machine_id), None)
        if not machine:
            self._state.last_action_error = f"Machine {assignment.machine_id} not found."
            return

        if machine.status != "idle":
            # If the agent assigned a job to a busy machine, ignore or record error
            # For now, we only allow assignments to idle machines.
            return

        if not assignment.job_id:
            # Idle action
            return

        job = next((j for j in self._state.jobs if j.job_id == assignment.job_id), None)
        if not job or job.status != "pending":
            self._state.last_action_error = f"Job {assignment.job_id} is not pending."
            return

        # Start processing
        setup_needed = machine.family is not None and machine.family != job.family
        setup_duration = self.CHANGEOVER_TIME if setup_needed else 0
        
        machine.status = "processing"
        machine.current_job_id = job.job_id
        machine.time_remaining = job.processing_time + setup_duration
        machine.family = job.family # Update family to the new one
        
        job.status = "processing"
        job.assigned_machine_id = machine.machine_id

    def _advance_time(self) -> None:
        """
        Advances the global clock and updates machine/job statuses.
        """
        # How much to advance?
        # If any machine is processing, advance to the earliest completion.
        # If no machine is processing, advance to the next job arrival or 1 unit.
        
        processing_machines = [m for m in self._state.machines if m.status == "processing"]
        pending_jobs = [j for j in self._state.jobs if j.status == "pending" and j.arrival_time > self._state.current_time]
        
        dt = 1
        if processing_machines:
            dt = min(m.time_remaining for m in processing_machines)
        elif pending_jobs:
            dt = min(j.arrival_time - self._state.current_time for j in pending_jobs)
        
        # Cap dt so we don't skip past max_time
        dt = min(dt, self._state.max_time - self._state.current_time)
        if dt <= 0:
            dt = 1 if self._state.current_time < self._state.max_time else 0

        if dt > 0:
            self._state.current_time += dt
            for machine in self._state.machines:
                if machine.status == "processing":
                    machine.time_remaining -= dt
                    
                    # Update job remaining time
                    job = next((j for j in self._state.jobs if j.job_id == machine.current_job_id), None)
                    if job:
                        job.remaining_time -= dt
                    
                    # Check completion
                    if machine.time_remaining <= 0:
                        machine.status = "idle"
                        machine.current_job_id = None
                        if job:
                            job.status = "completed"
                            job.completion_time = self._state.current_time
                            job.remaining_time = 0

    def state(self) -> EnvState:
        return self._state
