from .models import EnvState

def grade_episode(state: EnvState) -> float:
    """
    Computes a deterministic score for the episode in the 0.0-1.0 range.
    Based on total tardiness across all jobs.
    """
    total_tardiness = 0
    
    for job in state.jobs:
        if job.status == "completed":
            tardiness = max(0, job.completion_time - job.due_time)
            total_tardiness += tardiness
        else:
            # For jobs not completed, penalize based on current time or max time
            # This encourages the agent to finish all jobs.
            tardiness = max(0, state.current_time - job.due_time)
            # Add a penalty for not finishing
            total_tardiness += tardiness + 10 
            
    # Normalize score. 
    # For the easy scenario, we expect total tardiness to be relatively low.
    # Let's cap the normalization at 50 units of tardiness for a 0 score.
    tardiness_cap = 50.0
    score = 1.0 - (total_tardiness / tardiness_cap)
    
    return max(0.0, min(1.0, score))
