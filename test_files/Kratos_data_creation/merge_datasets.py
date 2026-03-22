"""
Dataset Merger — Combines multiple dataset folders into one.

Usage:
    python merge_datasets.py

Modes:
    "append"  — adds cases with new sequential IDs (no overwrite)
    "replace" — overwrites target folder entirely

Example:
    Sources: primal_Fx (100 cases), primal_Fz (100 cases), primal_combined (300 cases)
    Target:  primal (500 cases: case_primal_1 to case_primal_500)
"""

import os
import sys
import json
import shutil
from datetime import datetime
from typing import List, Dict, Tuple


# ============================================================
# CONFIGURATION — EDIT HERE
# ============================================================

WORKING_DIR = r"E:\Master_Thesis_Sumit_Mahindrakar\test_files\Kratos_data_creation"

# Source → Target mappings
MERGE_JOBS = [
    {
        "name": "Primal Merge",
        "sources": [
            {"folder": "primal_Fx", "prefix": "case_primal"},
            {"folder": "primal_Fz", "prefix": "case_primal"},
            {"folder": "primal_combined", "prefix": "case_primal"},
        ],
        "target_folder": "primal",
        "target_prefix": "case_primal",
        "mode": "append",  # "append" or "replace"
    },
    {
        "name": "Adjoint Merge",
        "sources": [
            {"folder": "adjoint_Fx", "prefix": "case_adjoint"},
            {"folder": "adjoint_Fz", "prefix": "case_adjoint"},
            {"folder": "adjoint_combined", "prefix": "case_adjoint"},
        ],
        "target_folder": "adjoint",
        "target_prefix": "case_adjoint",
        "mode": "append",  # "append" or "replace"
    },
]

# Set to True for dry run (no file operations)
DRY_RUN = False # True False


# ============================================================
# FUNCTIONS
# ============================================================

def find_existing_cases(folder: str, prefix: str) -> List[int]:
    """Find existing case IDs in a folder."""
    if not os.path.exists(folder):
        return []

    case_ids = []
    for name in os.listdir(folder):
        if name.startswith(prefix + "_") and os.path.isdir(
            os.path.join(folder, name)
        ):
            try:
                case_id = int(name.split("_")[-1])
                case_ids.append(case_id)
            except ValueError:
                continue

    return sorted(case_ids)


def find_source_cases(folder: str, prefix: str) -> List[Tuple[int, str]]:
    """Find source cases and return (original_id, full_path) pairs."""
    if not os.path.exists(folder):
        print(f"    ⚠ Source folder not found: {folder}")
        return []

    cases = []
    for name in sorted(os.listdir(folder)):
        if name.startswith(prefix + "_") and os.path.isdir(
            os.path.join(folder, name)
        ):
            try:
                case_id = int(name.split("_")[-1])
                full_path = os.path.join(folder, name)
                cases.append((case_id, full_path))
            except ValueError:
                continue

    return sorted(cases, key=lambda x: x[0])


def update_case_config(case_folder: str, new_id: int,
                       original_id: int, source_folder: str):
    """Update case_config.json with new ID and merge info."""
    config_path = os.path.join(case_folder, "case_config.json")
    if not os.path.exists(config_path):
        return

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        config["case_id"] = new_id
        config["merge_info"] = {
            "original_case_id": original_id,
            "source_folder": source_folder,
            "merged_timestamp": datetime.now().isoformat()
        }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"    ⚠ Could not update case_config.json: {e}")


def run_merge_job(job: Dict, working_dir: str, dry_run: bool = False):
    """Execute one merge job."""
    job_name = job["name"]
    sources = job["sources"]
    target_folder = os.path.join(working_dir, job["target_folder"])
    target_prefix = job["target_prefix"]
    mode = job["mode"]

    print(f"\n{'=' * 60}")
    print(f"MERGE JOB: {job_name}")
    print(f"{'=' * 60}")
    print(f"  Target: {target_folder}")
    print(f"  Prefix: {target_prefix}")
    print(f"  Mode:   {mode}")
    print(f"  Sources: {len(sources)}")

    for src in sources:
        src_path = os.path.join(working_dir, src["folder"])
        src_cases = find_source_cases(src_path, src["prefix"])
        print(f"    {src['folder']}: {len(src_cases)} cases")

    # ── Determine starting ID ──
    if mode == "replace":
        if os.path.exists(target_folder):
            if dry_run:
                print(f"  [DRY RUN] Would delete: {target_folder}")
            else:
                print(f"  Deleting existing target: {target_folder}")
                shutil.rmtree(target_folder)
        next_id = 1
    elif mode == "append":
        existing = find_existing_cases(target_folder, target_prefix)
        next_id = max(existing) + 1 if existing else 1
        print(f"  Existing cases: {len(existing)}")
        print(f"  Starting new ID: {next_id}")
    else:
        print(f"  ✗ Unknown mode: {mode}")
        return

    os.makedirs(target_folder, exist_ok=True)

    # ── Copy cases ──
    total_copied = 0
    merge_log = []

    for src in sources:
        src_path = os.path.join(working_dir, src["folder"])
        src_cases = find_source_cases(src_path, src["prefix"])
        src_name = src["folder"]

        if not src_cases:
            print(f"\n  Skipping {src_name} (no cases found)")
            continue

        print(f"\n  Copying from {src_name}...")

        for original_id, src_case_path in src_cases:
            new_folder_name = f"{target_prefix}_{next_id}"
            target_case_path = os.path.join(
                target_folder, new_folder_name
            )

            if dry_run:
                print(f"    [DRY RUN] {src_name}/{src['prefix']}_{original_id}"
                      f" → {new_folder_name}")
            else:
                shutil.copytree(src_case_path, target_case_path)
                update_case_config(
                    target_case_path, next_id,
                    original_id, src_name
                )

                if total_copied < 5 or total_copied % 50 == 0:
                    print(f"    {src_name}/{src['prefix']}_{original_id}"
                          f" → {new_folder_name}")

            merge_log.append({
                "new_id": next_id,
                "original_id": original_id,
                "source": src_name,
                "source_path": src_case_path,
                "target_path": target_case_path
            })

            next_id += 1
            total_copied += 1

    # ── Save merge log ──
    if not dry_run and merge_log:
        log_path = os.path.join(target_folder, "merge_log.json")

        # Append to existing log if present
        existing_log = []
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                existing_log = json.load(f)

        full_log = {
            "timestamp": datetime.now().isoformat(),
            "job_name": job_name,
            "mode": mode,
            "cases_added": total_copied,
            "total_cases": next_id - 1,
            "sources": [
                {
                    "folder": s["folder"],
                    "prefix": s["prefix"],
                    "cases": len(find_source_cases(
                        os.path.join(working_dir, s["folder"]),
                        s["prefix"]
                    ))
                }
                for s in sources
            ],
            "mapping": merge_log
        }

        if existing_log:
            if isinstance(existing_log, list):
                existing_log.append(full_log)
            else:
                existing_log = [existing_log, full_log]
            with open(log_path, 'w') as f:
                json.dump(existing_log, f, indent=4)
        else:
            with open(log_path, 'w') as f:
                json.dump(full_log, f, indent=4)

        print(f"\n  Merge log saved: {log_path}")

    # ── Summary ──
    print(f"\n  {'─' * 40}")
    print(f"  {job_name} COMPLETE")
    print(f"  Cases copied: {total_copied}")
    print(f"  Total in target: {next_id - 1}")
    if dry_run:
        print(f"  [DRY RUN — no files changed]")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("DATASET MERGER")
    print("=" * 60)
    print(f"Working directory: {WORKING_DIR}")
    print(f"Dry run: {DRY_RUN}")
    print(f"Jobs: {len(MERGE_JOBS)}")

    if not os.path.exists(WORKING_DIR):
        print(f"✗ Working directory not found: {WORKING_DIR}")
        sys.exit(1)

    # ── Ask for confirmation ──
    if not DRY_RUN:
        print(f"\n⚠ This will copy files. Proceed?")
        for job in MERGE_JOBS:
            mode_str = "REPLACE (delete existing!)" if job["mode"] == "replace" \
                else "APPEND"
            print(f"  {job['name']}: {mode_str} → {job['target_folder']}")

        response = input("\nType 'yes' to proceed: ").strip().lower()
        if response != 'yes':
            print("Cancelled.")
            sys.exit(0)

    # ── Run jobs ──
    for job in MERGE_JOBS:
        run_merge_job(job, WORKING_DIR, DRY_RUN)

    print(f"\n{'=' * 60}")
    print("ALL MERGE JOBS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()