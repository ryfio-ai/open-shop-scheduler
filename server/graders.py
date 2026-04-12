# server/graders.py - Standalone graders (no imports from envs)

_SCORE_MIN = 0.01
_SCORE_MAX = 0.99

def _safe_score(raw: float) -> float:
    """Clamp to strictly-open (0.01, 0.99) interval."""
    try:
        if not isinstance(raw, (int, float)):
            raw = float(raw)
        if raw != raw:  # NaN check
            raw = _SCORE_MIN
        if raw == float('inf'):
            raw = _SCORE_MAX
        if raw == float('-inf'):
            raw = _SCORE_MIN
        return max(_SCORE_MIN, min(_SCORE_MAX, float(raw)))
    except Exception:
        return _SCORE_MIN

def grade_episode(state) -> float:
    """Deterministic grader. Returns score strictly in (0.01, 0.99)."""
    if state is None:
        return _safe_score(_SCORE_MIN)
    
    if isinstance(state, dict):
        jobs = state.get("jobs", []) or state.get("completed_jobs", [])
        current_time = state.get("current_time", 0)
        
        if not jobs:
            return _safe_score(_SCORE_MIN)
        
        total_tardiness = 0.0
        for job in jobs:
            if isinstance(job, dict):
                comp_time = job.get("completion_time")
                due_time = job.get("due_time") if job.get("due_time") is not None else job.get("due_date", 1)
                if due_time <= 0:
                    due_time = 1
            else:
                comp_time = getattr(job, "completion_time", None)
                due_time = getattr(job, "due_time", None) or getattr(job, "due_date", 1)
                if due_time <= 0:
                    due_time = 1
            
            if comp_time is not None:
                total_tardiness += max(0.0, float(comp_time) - float(due_time))
            else:
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
                tardiness = max(0.0, job.completion_time - getattr(job, 'due_date', 1))
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
    
    return _safe_score(_SCORE_MIN)

def grade_easy_single_machine(episode_state: dict) -> float:
    return grade_episode(episode_state)

def grade_medium_parallel_changeover(episode_state: dict) -> float:
    return grade_episode(episode_state)

def grade_hard_dynamic_arrivals(episode_state: dict) -> float:
    return grade_episode(episode_state)

__all__ = [
    "grade_easy_single_machine",
    "grade_medium_parallel_changeover",
    "grade_hard_dynamic_arrivals",
    "grade_episode",
    "_SCORE_MIN",
    "_SCORE_MAX"
]
