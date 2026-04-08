---
title: OpenShop Scheduler
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
tags:
  - openenv
  - manufacturing
  - scheduling
---

# OpenShop Scheduler (OpenEnv)

## 🎯 Environment Motivation
Manufacturing production scheduling is a high-stakes, real-world task where AI agents can drive significant efficiency. In a "Job Shop" or "Open Shop" setting, agents must allocate limited machine resources to competing jobs while minimizing **tardiness** (delay beyond due dates) and managing **setup costs** (time lost when switching job families). 

Unlike simple games, this environment models the complexities of factory throughput, including dynamic job arrivals and family-based changeover penalties.

## 🕹️ Observation Space
The `Observation` is a Pydantic model containing:
- `task_id`: Current task identifier.
- `current_time`: The global clock (units of time).
- `machines`: List of `MachineSnapshot` (status, current job, family, time remaining).
- `jobs_pending`: List of `JobSnapshot` for jobs that have arrived and are ready for assignment.
- `jobs_in_progress`: Jobs currently being processed.
- `completed_jobs`: History of finished tasks with completion times.
- `last_action_error`: Feedback if the previous step was invalid.

## 🛠️ Action Space
The `Action` is a Pydantic model:
- `assignments`: List of `MachineAssignment` objects.
  - `machine_id`: The machine to assign work to.
  - `job_id`: The job to start (must be in `jobs_pending`).
- `reasoning`: A string field for the agent to explain its scheduling logic.

## 🏆 Tasks & Difficulty
We provide 3 standard tasks with increasing complexity:
1. **`easy_single_machine`**: 5 jobs on 1 machine. Focuses on basic sequencing.
2. **`medium_parallel_changeover`**: 6 jobs on 2 parallel machines. Introduces 2-unit changeover penalties when switching families.
3. **`hard_dynamic_arrivals`**: 8 jobs on 3 machines. Jobs arrive at different times ($t=0$ to $t=40$), requiring real-time adaptation.

## 📊 Evaluation & Rewards
- **Graders**: Deterministic success criteria based on total tardiness. Score is in `[0.0, 1.0]`.
- **Reward Shaping**: Includes completion bonuses, penalty for tardiness, and heuristic bonuses for following Earliest Due Date (EDD) principles.

## 🔧 Setup & Usage

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
