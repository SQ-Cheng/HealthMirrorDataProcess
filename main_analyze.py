#!/usr/bin/env python
"""Unified entry point for data analysis and visualization tasks.

Subcommands:
    ptt-bp          Batch PTT extraction for BP correlation (ptt_extract.py)
    ptt-bp-explore  Exploratory PTT-BP correlation analysis (bp_ptt_explore.py)
    hrv-stim        HRV analysis for electrical-stimulation patients (calc_stim_hrv.py)
    bp-dist         Blood pressure distribution histograms (visualize_bp_distribution.py)
    bp-ptt          Interactive PTT vs BP scatter with filters (visualize_bp_ptt.py)
    bp-ptt-single   Per-patient PTT vs BP visualization (visualize_bp_ptt_single.py)
    sample-stats    Report cleaned sample lengths (report_sample_lengths.py)
    count-points    Count raw/cleaned data points (count_data_points.py)
    find-stim       Find electrical stimulation patients (find_electrical_stimulation.py)
"""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COMMANDS = {
    "ptt-bp": {
        "script": "ptt_extract.py",
        "help": "Batch PTT extraction from cleaned data for BP patients",
    },
    "ptt-bp-explore": {
        "script": "bp_ptt_explore.py",
        "help": "Exploratory PTT-to-BP correlation with threshold optimization",
    },
    "hrv-stim": {
        "script": "calc_stim_hrv.py",
        "help": "HRV analysis for electrical-stimulation patients",
    },
    "bp-dist": {
        "script": "visualize_bp_distribution.py",
        "help": "Blood pressure distribution histograms",
    },
    "bp-ptt": {
        "script": "visualize_bp_ptt.py",
        "help": "Interactive PTT vs BP scatter plot with quality filters",
    },
    "bp-ptt-single": {
        "script": "visualize_bp_ptt_single.py",
        "help": "Per-patient interactive PTT vs BP visualization",
    },
    "sample-stats": {
        "script": "report_sample_lengths.py",
        "help": "Report sample lengths across cleaning output directories",
    },
    "count-points": {
        "script": "count_data_points.py",
        "help": "Count raw and cleaned data points for a patient range",
    },
    "find-stim": {
        "script": "find_electrical_stimulation.py",
        "help": "Scan patient_info.txt files for electrical-stimulation patients",
    },
}


def print_help():
    print("usage: main_analyze.py <command> [<args>...]")
    print()
    print("Data analysis and visualization tools")
    print()
    print("commands:")
    for name, info in COMMANDS.items():
        print(f"  {name:<18} {info['help']}")
    print()
    print("Run 'main_analyze.py <command> --help' for command-specific options.")


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
