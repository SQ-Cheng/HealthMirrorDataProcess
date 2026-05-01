import argparse
from data.patient_info import PatientInfo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patient info from patient directories")
    parser.add_argument("--mirror-id", type=int, default=6, help="Mirror ID for data directory naming")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory (overrides --mirror-id based default)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file (default: overall_patient_info_{mirror_id}.csv)")
    parser.add_argument("--mode", type=str, default="dir", choices=["dir", "file"], help="Extraction mode")
    parser.add_argument("--data-file", type=str, default=None, help="Data file for file mode")
    args = parser.parse_args()

    data_dir = args.data_dir or f"./mirror{args.mirror_id}_data"
    output_file = args.output or f"overall_patient_info_{args.mirror_id}.csv"

    patient_info = PatientInfo(data_dir, save_dir=output_file, mode=args.mode)
    if args.data_file:
        patient_info.extract(data_file=args.data_file)
    else:
        patient_info.extract()
    patient_info.save()
