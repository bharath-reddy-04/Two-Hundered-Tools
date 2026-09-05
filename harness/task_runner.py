import json

from agent import run_agent
from verifier import verify_task


def load_tasks(path):
    with open(path, "r") as f:
        return json.load(f)


def run_all_tasks():
    tasks = load_tasks("tasks.json")

    results = []

    for task in tasks:
        print(f"\nRunning {task['task_id']}...")
        
        # 1. Give task to the agent
        agent_result = run_agent(task["instruction"])

        # 2. Verify actual sandbox state
        verification = verify_task(task, agent_result)

        result = {
            "task_id": task["task_id"],
            "agent_result": agent_result,
            "verification": verification,
            "passed": verification["passed"]
        }

        results.append(result)

        print("PASS" if result["passed"] else "FAIL")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_all_tasks()