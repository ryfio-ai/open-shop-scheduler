"""
Bulletproof graders for Open Shop Scheduler.
Handles ALL edge cases to ensure score is always in (0.01, 0.99).
"""

# Safe bounds - strictly between 0 and 1
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def _safe_score(raw) -> float:
    """
    Clamp to strictly-open (0, 1) interval.
    Handles: None, NaN, Inf, negative values, values > 1
    """
    try:
        if raw is None:
            return _SCORE_MIN
        raw = float(raw)
        if raw != raw:  # NaN check
            return _SCORE_MIN
        if raw == float('inf'):
            return _SCORE_MAX
        if raw == float('-inf'):
            return _SCORE_MIN
        # Clamp to valid range
        return max(_SCORE_MIN, min(_SCORE_MAX, raw))
    except Exception:
        # Any error - return safe default
        return _SCORE_MIN


def grade_episode(state) -> float:
    """
    Grade an episode. Returns score strictly in (0.01, 0.99).
    Handles: None, empty state, missing keys, wrong types
    """
    # Handle None state
    if state is None:
        return _SCORE_MIN

    # Handle dict state (from API)
    if isinstance(state, dict):
        jobs = state.get("jobs") or state.get("completed_jobs") or []
        current_time = state.get("current_time", 0)

        if not jobs:
            return _SCORE_MIN

        total_tardiness = 0.0
        valid_jobs = 0

        for job in jobs:
            if not isinstance(job, dict):
                continue

            comp_time = job.get("completion_time")
            if comp_time is None:
                continue

            # Get due time with proper None handling
            due_date = job.get("due_date")
            due_time_val = job.get("due_time")

            if due_date is not None:
                due_time = float(due_date)
            elif due_time_val is not None:
                due_time = float(due_time_val)
            else:
                due_time = 1.0

            if due_time <= 0:
                due_time = 1.0

            tardiness = max(0.0, float(comp_time) - due_time)
            total_tardiness += tardiness
            valid_jobs += 1

        if valid_jobs == 0:
            return _SCORE_MIN

        # Calculate score
        tardiness_cap = 20.0 * valid_jobs
        raw_score = 1.0 - (total_tardiness / tardiness_cap)
        return _safe_score(raw_score)

    # Handle object state (EnvState)
    if hasattr(state, 'jobs'):
        jobs = state.jobs
        if not jobs:
            return _SCORE_MIN

        total_tardiness = 0.0
        valid_jobs = 0

        for job in jobs:
            if hasattr(job, 'completion_time') and job.completion_time is not None:
                due = getattr(job, 'due_date', 1)
                if due <= 0:
                    due = 1
                tardiness = max(0.0, job.completion_time - due)
                total_tardiness += tardiness
                valid_jobs += 1
            elif hasattr(job, 'status'):
                # Incomplete job penalty
                current_t = getattr(state, 'current_time', 0)
                due = getattr(job, 'due_date', 1)
                if due <= 0:
                    due = 1
                overdue = max(0.0, current_t - due)
                total_tardiness += overdue + 15.0
                valid_jobs += 1

        if valid_jobs == 0:
            return _SCORE_MIN

        tardiness_cap = 20.0 * valid_jobs
        raw_score = 1.0 - (total_tardiness / tardiness_cap)
        return _safe_score(raw_score)

    # Unknown state type - return safe default
    return _SCORE_MIN


# Task-specific graders
def grade_easy_single_machine(episode_state=None) -> float:
    """Grade easy task"""
    return grade_episode(episode_state)


def grade_medium_parallel_changeover(episode_state=None) -> float:
    """Grade medium task"""
    return grade_episode(episode_state)


def grade_hard_dynamic_arrivals(episode_state=None) -> float:
    """Grade hard task"""
    return grade_episode(episode_state)


# Unified entry point
def grade(task_id: str = "", episode_state=None) -> float:
    """Unified grader entry point"""
    return grade_episode(episode_state)
