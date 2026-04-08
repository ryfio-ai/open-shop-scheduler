---
title: OpenShop Scheduler
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
---

# OpenShop Scheduler (OpenEnv)

A manufacturing shop-floor scheduling environment built for the **OpenEnv** benchmark standard. This repo provides a verifiable environment for training and evaluating AI agents on production scheduling tasks.

## Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional)
- OpenAI API Key (or compatible endpoint)

### Local Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables:
   ```bash
   export API_BASE_URL="https://api.openai.com/v1"
   export MODEL_NAME="gpt-4"
   export HF_TOKEN="your-api-key"
   ```
4. Run inference:
   ```bash
   python inference.py --task_id easy_single_machine
   ```

## Task Scenarios

1. **`easy_single_machine`**: A baseline task with 5 jobs and 1 machine. Focuses on basic tardiness minimization.
2. **`medium_parallel_changeover`**: 2 machines. Jobs belong to different "families". Switching families on a machine incurs a **setup time** penalty.
3. **`hard_dynamic_arrivals`**: 3 machines and 8 jobs. Jobs have different **arrival times**. The agent must handle jobs that appear mid-episode.

## Environment API

The environment follows the OpenEnv standard with Pydantic-typed observations and actions.

### Observation Space
- `current_time`: Global clock.
- `machines`: List of machine states (idle/processing, family, time remaining).
- `jobs_pending`: Jobs that have arrived and are waiting for assignment.
- `jobs_in_progress`: Jobs currently being processed.
- `completed_jobs`: Finished jobs with their completion stats.

### Action Space
- `assignments`: A list of `(machine_id, job_id)` pairs to start processing.
- `reasoning`: (Optional) String explaining the agent's decision.

## Docker Usage

Build the image:
```bash
docker build -t shop-scheduler .
```

Run a task:
```bash
docker run -e API_BASE_URL=$API_BASE_URL -e MODEL_NAME=$MODEL_NAME -e HF_TOKEN=$HF_TOKEN shop-scheduler --task_id medium_parallel_changeover
```

## Logging Format

The `inference.py` script outputs standard logs for automated evaluation:
- `[START]`: Task and model metadata.
- `[STEP]`: Action taken, reward received, and per-step status.
- `[END]`: Final success status, score [0-1], and full reward history.
