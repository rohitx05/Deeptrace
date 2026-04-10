"""Project memory command interface."""

import argparse
import json

from utils.project_memory import ProjectMemory, apply_dotted_update, parse_update_value


def parse_args():
    parser = argparse.ArgumentParser(description="Persistent project memory commands")
    parser.add_argument("command", choices=["load_state", "update_state", "compress_context", "next_step"])
    parser.add_argument("--json", dest="json_payload", type=str, default=None)
    parser.add_argument("--set", dest="set_values", action="append", default=[])
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--last_step", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    memory = ProjectMemory()

    if args.command == "load_state":
        state, _ = memory.load_primary_context()
        payload = {
            "last_completed_step": state.get("last_completed_step"),
            "active_checkpoint": state.get("checkpoint_paths", {}).get("active"),
            "active_dataset": state.get("dataset_info", {}).get("active_dataset"),
            "accuracy": state.get("performance_metrics", {}).get("accuracy"),
            "auc": state.get("performance_metrics", {}).get("auc"),
            "optimal_threshold": state.get("performance_metrics", {}).get("optimal_threshold"),
            "temperature": state.get("calibration_status", {}).get("temperature"),
            "calibration_status": state.get("calibration_status", {}).get("status"),
            "notes": state.get("notes"),
        }
        print(json.dumps(payload, indent=2))
        return

    if args.command == "update_state":
        updates = {}
        if args.json_payload:
            updates = json.loads(args.json_payload)
        for item in args.set_values:
            key, value = item.split("=", 1)
            apply_dotted_update(updates, key, parse_update_value(value))
        state = memory.update_state(updates, notes=args.notes, last_step=args.last_step)
        print(json.dumps({
            "status": "updated",
            "last_completed_step": state.get("last_completed_step"),
            "updated_at": state.get("project", {}).get("updated_at"),
        }, indent=2))
        return

    if args.command == "compress_context":
        summary = memory.compress_context()
        print(summary)
        return

    if args.command == "next_step":
        memory.load_primary_context()
        print(memory.suggest_next_step())


if __name__ == "__main__":
    main()
