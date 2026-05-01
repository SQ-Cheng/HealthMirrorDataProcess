#!/usr/bin/env python
"""Unified entry point for data washing and cleaning tasks.

Subcommands:
    auto          Batch auto-wash with SQI filtering (auto_wash.py)
    interactive   Interactive slider-based review (wash_data.py)
    slice         Slice recordings into fixed-duration segments (data_slicer.py)
"""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COMMANDS = {
    "auto": {
        "script": "auto_wash.py",
        "help": "Batch auto-wash with configurable SQI method",
    },
    "interactive": {
        "script": "wash_data.py",
        "help": "Interactive slider-based signal review and cleaning",
    },
    "slice": {
        "script": "data_slicer.py",
        "help": "Slice raw recordings into fixed-duration segments",
    },
}


def print_help():
    print("usage: main_wash.py <command> [<args>...]")
    print()
    print("Data washing and cleaning tools")
    print()
    print("commands:")
    for name, info in COMMANDS.items():
        print(f"  {name:<16} {info['help']}")
    print()
    print("Run 'main_wash.py <command> --help' for command-specific options.")


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
