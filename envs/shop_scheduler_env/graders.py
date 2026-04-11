from .models import EnvState

# Safe bounds that satisfy the validator's strict (0.0, 1.0) exclusive check
# Matching the pattern used by passing submissions: floor=0.01, ceiling=0.99
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def _safe_score(raw: float) -> float:
    """Clamp to strictly-open (0, 1) interval. Never returns 0.0 or 1.0."""
    return max(_SCORE_MIN, min(_SCORE_MAX, float(raw)))


def grade_episode(state) -> float:
    """
    Deterministic grader. Returns a score strictly in (0.01, 0.99).

    Strategy: penalise tardiness across all jobs.
    - Completed jobs: tardiness = max(0, completion_time - due_time)
    - Incomplete jobs: heavy flat penalty to push agent to finish everything
    """
    if isinstance(state, dict):
        # Handle different key names from episode_state
        jobs = state.get("jobs", []) or state.get("completed_jobs", [])
        current_time = state.get("current_time", 0)
        
        total_tardiness = 0.0
        for job in jobs:
            # Handle both object and dict access
            if isinstance(job, dict):
                comp_time = job.get("completion_time")
                due_time = job.get("due_date") or job.get("due_time", 1)
            else:
                comp_time = getattr(job, "completion_time", None)
                due_time = getattr(job, "due_date", getattr(job, "due_time", 1))

            if comp_time is not None:
                total_tardiness += max(0.0, float(comp_time) - float(due_time))
            else:
                # Penalty for incomplete job
                total_tardiness += max(0.0, float(current_time) - float(due_time)) + 15.0

        total_jobs = len(jobs) or 1
        tardiness_cap = 20.0 * total_jobs
        raw = 1.0 - (total_tardiness / tardiness_cap)
        return _safe_score(raw)

    # Original model-based logic
    total_jobs = len(state.jobs)
    if total_jobs == 0:
        return _SCORE_MIN

    total_tardiness = 0.0
    for job in state.jobs:
        if job.status == "completed" and job.completion_time is not None:
            tardiness = max(0.0, job.completion_time - job.due_date)
            total_tardiness += tardiness
        else:
            overdue = max(0.0, state.current_time - job.due_date)
            total_tardiness += overdue + 15.0

    tardiness_cap = 20.0 * total_jobs
    raw = 1.0 - (total_tardiness / tardiness_cap)
    return _safe_score(raw)

def grade_easy_single_machine(episode_state: dict) -> float:
    """Grade easy task - must return float in (0.01, 0.99)"""
    return grade_episode(episode_state)

def grade_medium_parallel_changeover(episode_state: dict) -> float:
    """Grade medium task"""
    return grade_episode(episode_state)

def grade_hard_dynamic_arrivals(episode_state: dict) -> float:
    """Grade hard task"""
    return grade_episode(episode_state)
