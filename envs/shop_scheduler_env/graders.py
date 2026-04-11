from .models import EnvState

# Safe bounds that satisfy the validator's strict (0.0, 1.0) exclusive check
# MUST be strictly between 0 and 1, not including boundaries
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99

def _safe_score(raw: float) -> float:
    """
    Clamp to strictly-open (0, 1) interval. Never returns 0.0 or 1.0.
    Also handles NaN and Inf cases.
    """
    if not isinstance(raw, (int, float)):
        raw = float(raw)
    if raw != raw:  # Check for NaN (NaN != NaN)
        raw = _SCORE_MIN
    if raw == float('inf'):
        raw = _SCORE_MAX
    if raw == float('-inf'):
        raw = _SCORE_MIN
    raw = max(_SCORE_MIN, min(_SCORE_MAX, float(raw)))
    return raw

def grade_episode(state) -> float:
    """
    Deterministic grader. Returns a score strictly in (0.01, 0.99).
    """
    # Handle None or invalid state
    if state is None:
        return _SCORE_MIN
    
    if isinstance(state, dict):
        # Handle different key names from episode_state
        jobs = state.get("jobs", []) or state.get("completed_jobs", [])
        current_time = state.get("current_time", 0)
        
        # CRITICAL: If no jobs, return minimum valid score (not 0.0)
        if not jobs:
            return _SCORE_MIN
        
        total_tardiness = 0.0
        for job in jobs:
            if isinstance(job, dict):
                comp_time = job.get("completion_time")
                
                # FIX: Handle due_date=0 correctly - don't use 'or' operator!
                due_date = job.get("due_date")
                due_time_val = job.get("due_time")
                
                # Explicit None check to avoid 0 being treated as falsy
                if due_date is not None:
                    due_time = due_date
                elif due_time_val is not None:
                    due_time = due_time_val
                else:
                    due_time = 1  # Default fallback
                
                # Prevent division by zero or negative due times
                if due_time <= 0:
                    due_time = 1
            else:
                comp_time = getattr(job, "completion_time", None)
                due_time = getattr(job, "due_date", None)
                if due_time is None:
                    due_time = getattr(job, "due_time", 1)
                if due_time <= 0:
                    due_time = 1

            if comp_time is not None:
                total_tardiness += max(0.0, float(comp_time) - float(due_time))
            else:
                # Penalty for incomplete job
                total_tardiness += max(0.0, float(current_time) - float(due_time)) + 15.0

        total_jobs = len(jobs)
        tardiness_cap = 20.0 * max(total_jobs, 1)
        raw = 1.0 - (total_tardiness / tardiness_cap)
        return _safe_score(raw)

    if hasattr(state, 'jobs') and state.jobs:
        total_jobs = len(state.jobs)
        total_tardiness = 0.0
        for job in state.jobs:
            if hasattr(job, 'status') and job.status == "completed" and hasattr(job, 'completion_time') and job.completion_time is not None:
                tardiness = max(0.0, job.completion_time - job.due_date)
                total_tardiness += tardiness
            else:
                current_t = getattr(state, 'current_time', 0)
                due = getattr(job, 'due_date', 1)
                if due <= 0:
                    due = 1
                overdue = max(0.0, current_t - due)
                total_tardiness += overdue + 15.0

        tardiness_cap = 20.0 * total_jobs
        raw = 1.0 - (total_tardiness / tardiness_cap)
        return _safe_score(raw)

    # Added fallback for unexpected types or empty states
    print(f"[DEBUG] Unknown state type or empty: {type(state)}", flush=True)
    return _safe_score(0.5)

def grade_easy_single_machine(episode_state: dict) -> float:
    """Grade easy task - must return float in (0.01, 0.99)"""
    return grade_episode(episode_state)

def grade_medium_parallel_changeover(episode_state: dict) -> float:
    """Grade medium task"""
    return grade_episode(episode_state)

def grade_hard_dynamic_arrivals(episode_state: dict) -> float:
    """Grade hard task"""
    return grade_episode(episode_state)
