#!/usr/bin/env python
"""Unified entry point for patient info management tasks.

Subcommands:
    extract    Extract patient info from data directories (extract_patient_info.py)
    merge      Merge two patient info CSV files (merge_patient_info.py)
    merge-lab  Merge lab test XLSX data into merged patient info CSV (merge_lab_xlsx.py)
"""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COMMANDS = {
    "extract": {
        "script": "extract_patient_info.py",
        "help": "Extract patient info from patient directories",
    },
    "merge": {
        "script": "merge_patient_info.py",
        "help": "Merge extracted and marked patient info CSV files",
    },
    "merge-lab": {
        "script": "merge_lab_xlsx.py",
        "help": "Merge lab test XLSX data into merged patient info CSV",
    },
}


def print_help():
    print("usage: main_patient.py <command> [<args>...]")
    print()
    print("Patient info management tools")
    print()
    print("commands:")
    for name, info in COMMANDS.items():
        print(f"  {name:<12} {info['help']}")
    print()
    print("Run 'main_patient.py <command> --help' for command-specific options.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print_help()
        sys.exit(0)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}")
        print_help()
        sys.exit(1)

    script = os.path.join(SCRIPT_DIR, COMMANDS[command]["script"])
    cmd = [sys.executable, script] + sys.argv[2:]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
