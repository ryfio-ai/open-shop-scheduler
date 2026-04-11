from .models import EnvState

# Safe bounds that satisfy the validator's strict (0.0, 1.0) exclusive check
# Matching the pattern used by passing submissions: floor=0.001, ceiling=0.999
_SCORE_MIN = 0.001
_SCORE_MAX = 0.999


def _safe_score(raw: float) -> float:
    """Clamp to strictly-open (0, 1) interval. Never returns 0.0 or 1.0."""
    return max(_SCORE_MIN, min(_SCORE_MAX, float(raw)))


def grade_episode(state) -> float:
    """
    Deterministic grader. Returns a score strictly in (0.001, 0.999).

    Strategy: penalise tardiness across all jobs.
    - Completed jobs: tardiness = max(0, completion_time - due_time)
    - Incomplete jobs: heavy flat penalty to push agent to finish everything
    """
    if isinstance(state, dict):
        from .models import EnvState as _ES
        state = _ES.model_validate(state)

    total_jobs = len(state.jobs)
    if total_jobs == 0:
        return _SCORE_MIN

    total_tardiness = 0.0
    incomplete_count = 0

    for job in state.jobs:
        if job.status == "completed" and job.completion_time is not None:
            tardiness = max(0.0, job.completion_time - job.due_time)
            total_tardiness += tardiness
        else:
            incomplete_count += 1
            # Penalty per incomplete job = remaining time + overdue gap
            overdue = max(0.0, state.current_time - job.due_time)
            total_tardiness += overdue + 15.0   # 15-unit flat penalty per unfinished job

    # Scale the cap to the number of jobs so score is fair across all task sizes
    # Cap = 20 units of tardiness per job → all late by 20 units = score ≈ 0.001
    tardiness_cap = 20.0 * total_jobs
    if tardiness_cap <= 0:
        tardiness_cap = 1.0

    raw = 1.0 - (total_tardiness / tardiness_cap)
    return _safe_score(raw)
