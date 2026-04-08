from .models import EnvState, Action, Reward, JobStatus
from typing import List, Set

def compute_step_reward(state: EnvState, action: Action) -> Reward:
    """
    Computes a shaped reward for the current step.
    """
    reward_val = 0.0
    on_time_bonus = 0.0
    tardiness_penalty = 0.0
    invalid_action_penalty = 0.0
    
    # 1. Check for invalid actions
    assigned_job_ids = [a.job_id for a in action.assignments if a.job_id is not None]
    
    # Check for duplicates or invalid job IDs
    pending_job_ids = {j.job_id for j in state.jobs if j.status == "pending"}
    
    for job_id in assigned_job_ids:
        if job_id not in pending_job_ids:
            invalid_action_penalty += 0.20
            reward_val -= 0.20
            
    # 2. Heuristic bonus: reward sequence if it follows earliest due date (EDD)
    if assigned_job_ids:
        pending_jobs = [j for j in state.jobs if j.status == "pending"]
        if pending_jobs:
            # Sort pending jobs by due date
            edd_sorted = sorted(pending_jobs, key=lambda x: x.due_time)
            top_2_edd = {j.job_id for j in edd_sorted[:2]}
            
            for job_id in assigned_job_ids:
                if job_id in top_2_edd:
                    reward_val += 0.20
    
    # 3. Completion bonus and tardiness penalty
    # This is checked by inspecting which jobs just finished in the last tick
    # (The environment logic will update jobs, but the reward script needs to see the transition)
    # However, usually reward is computed *after* the state transition in the step() function.
    # So we look at jobs that just reached 'completed'.
    
    for job in state.jobs:
        if job.status == "completed" and job.completion_time == state.current_time:
            # Job just completed in this step
            if job.completion_time <= job.due_time:
                on_time_bonus += 0.40
                reward_val += 0.40
            else:
                tardiness = job.completion_time - job.due_time
                tardiness_penalty += 0.10 * (tardiness / 10.0) # Scaled penalty
                reward_val -= 0.10

    # Ensure reward is in [0, 1] for the 'value' field if required, 
    # but the prompt says 'value: float = Field(ge=0.0, le=1.0)'.
    # If it's negative, we clamp it to 0 for the 'value' but keep components.
    
    return Reward(
        value=max(0.0, min(1.0, reward_val)),
        on_time_bonus=on_time_bonus,
        tardiness_penalty=tardiness_penalty,
        invalid_action_penalty=invalid_action_penalty,
        info={
            "raw_reward": reward_val
        }
    )
