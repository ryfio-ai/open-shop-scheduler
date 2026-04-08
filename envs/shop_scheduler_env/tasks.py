from pathlib import Path
import json
from typing import Dict, Any

def load_scenario(task_id: str) -> Dict[str, Any]:
    """
    Maps task IDs to scenario JSON files and returns the scenario data.
    """
    base_dir = Path(__file__).parent
    
    mapping = {
        "easy_single_machine": "scenarios/task_easy_single_machine.json",
        "medium_parallel_changeover": "scenarios/task_medium_parallel_changeover.json",
        "hard_dynamic_arrivals": "scenarios/task_hard_dynamic_arrivals.json",
    }
    
    if task_id not in mapping:
        raise ValueError(f"Task ID '{task_id}' not found in scenario mapping.")
    
    path = base_dir / mapping[task_id]
    
    if not path.exists():
        raise FileNotFoundError(f"Scenario file for '{task_id}' not found at {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
