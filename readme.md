# HealthMirror rPPG Data Processing

Unified data processing pipeline for HealthMirror rPPG (remote photoplethysmography) research — from raw recording to analysis-ready datasets.

## Entry Points

Three unified CLIs consolidate the project's 80+ scripts into focused task groups.

- **`main_wash.py`** — Signal washing, quality filtering, and slicing
- **`main_analyze.py`** — PTT/HRV analysis, BP visualization, data statistics
- **`main_patient.py`** — Patient info extraction and CSV merging

All entry points delegate to the underlying scripts and preserve their exit codes. Use `--help` on any command to see its full flags.

---

## `main_wash.py` — Data Washing & Cleaning

Three subcommands for preparing raw recordings.

| Subcommand | Script | Purpose |
|-----------|--------|---------|
| `auto` | `auto_wash.py` | Batch auto-wash with configurable SQI method |
| `interactive` | `wash_data.py` | Slider-based manual review and cleaning |
| `slice` | `data_slicer.py` | Slice recordings into fixed-duration segments |

### `auto` — Batch auto-wash

Processes patient folders through an SQI (Signal Quality Index) pipeline: segments the recording into windows, scores each window's ECG and rPPG quality, and outputs cleaned segments above threshold.

Two SQI methods:
- **`reference`** (default): NeuroKit2-based ECG + template-matching rPPG quality
- **`fused`**: Multi-metric fusion (autocorrelation, beat-to-beat correlation, template similarity, SNR) with configurable per-metric weights

```bash
# Basic usage — reference SQI, mirror version 1
python main_wash.py auto --data_dir ./patient_data --output_dir ./cleaned_data

# Fused SQI with custom thresholds and weights
python main_wash.py auto \
    --data_dir ./patient_data \
    --output_dir ./cleaned_sqi \
    --sqi_method fused \
    --threshold_ecg 0.7 \
    --threshold_rppg 0.6 \
    --ecg_weight_btb_corr 0.4 \
    --ecg_weight_template 0.3 \
    --ecg_weight_autocorr 0.3 \
    --rppg_weight_snr 0.6 \
    --rppg_weight_autocorr 0.4

# Reference SQI with visualization for manual review
python main_wash.py auto \
    --data_dir ./patient_data \
    --output_dir ./cleaned_data \
    --sqi_method reference \
    --visualize \
    --mirror_version 2

# Specify reference signal directory and patient info
python main_wash.py auto \
    --data_dir ./mirror4_data \
    --output_dir ./mirror4_cleaned \
    --reference_dir ./references \
    --patient_info_csv ./patient_bp.csv
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | (auto) | Patient data directory |
| `--output_dir` | (auto) | Output for cleaned segments |
| `--reference_dir` | (auto) | Reference signal directory |
| `--sqi_method` | `reference` | `reference` or `fused` |
| `--threshold_ecg` | varies | ECG SQI pass threshold |
| `--threshold_rppg` | varies | rPPG SQI pass threshold |
| `--visualize` | off | Enable per-patient review plots |
| `--mirror_version` | `1` | `1` (mirror1/2) or `2` (mirror4/5/6) |

Fused-SQI-specific flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--ecg_weight_autocorr` | 0.2 | Autocorrelation weight (ECG) |
| `--ecg_weight_btb_corr` | 0.4 | Beat-to-beat correlation weight (ECG) |
| `--ecg_weight_template` | 0.4 | Template matching weight (ECG) |
| `--rppg_weight_snr` | 0.5 | Frequency-domain SNR weight (rPPG) |
| `--rppg_weight_autocorr` | 0.5 | Autocorrelation weight (rPPG) |
| `--polarity` | `neg` | ECG signal polarity |

### `interactive` — Slider-based manual review

Opens an interactive matplotlib interface with sliders for threshold adjustment. Useful for spot-checking cleaning parameters on a subset of patients.

```bash
# Full interactive session
python main_wash.py interactive --data-path ./patient_data --log-path ./wash_logs

# Restrict to patient range
python main_wash.py interactive --start 0 --end 10
```

### `slice` — Fixed-duration segmentation

Slices raw recordings into uniform-length segments at a target sample rate.

```bash
# 10-second segments at 512 Hz
python main_wash.py slice \
    --input ./patient_data \
    --output ./sliced_data \
    --segment-duration 10.0 \
    --target-fs 512

# Process a patient subset
python main_wash.py slice --start 100 --end 200
```

---

## `main_analyze.py` — Data Analysis & Visualization

Nine subcommands for PTT/HRV/BP analysis and data reporting.

| Subcommand | Script | Purpose |
|-----------|--------|---------|
| `ptt-bp` | `ptt_extract.py` | Batch PTT extraction for BP correlation |
| `ptt-bp-explore` | `bp_ptt_explore.py` | Exploratory PTT-BP correlation with threshold opt. |
| `hrv-stim` | `calc_stim_hrv.py` | HRV analysis for electrical-stimulation patients |
| `bp-dist` | `visualize_bp_distribution.py` | SBP/DBP distribution histograms |
| `bp-ptt` | `visualize_bp_ptt.py` | Interactive PTT vs BP scatter with quality filters |
| `bp-ptt-single` | `visualize_bp_ptt_single.py` | Per-patient PTT vs BP visualization |
| `sample-stats` | `report_sample_lengths.py` | Report cleaned sample lengths |
| `count-points` | `count_data_points.py` | Count raw/cleaned data points |
| `find-stim` | `find_electrical_stimulation.py` | Find electrical stimulation patients |

### `ptt-bp` — Batch PTT extraction

Extracts Pulse Transit Time from cleaned data for BP correlation studies.

```bash
python main_analyze.py ptt-bp
```

### `hrv-stim` — Stimulation HRV analysis

Calculates HRV (RMSSD) for electrical-stimulation patients.

```bash
# Run analysis
python main_analyze.py hrv-stim

# With interactive visualization
python main_analyze.py hrv-stim --vis
```

### `bp-dist` — BP distribution histograms

Plots SBP and DBP distributions from merged patient info CSVs.

```bash
# Use default CSV directory
python main_analyze.py bp-dist --csv-dir ./csvs --bins 30 --save bp_hist.png

# Show without saving
python main_analyze.py bp-dist --no-show
```

### `bp-ptt` — Interactive PTT vs BP scatter

Interactive scatter plot of PTT against SBP/DBP with quality-based filtering.

```bash
python main_analyze.py bp-ptt \
    --csv ./cleaned_patient_info.csv \
    --ptt_stddev_max 0.1 \
    --ecg_sqi_min 0.7 \
    --rppg_sqi_min 0.8
```

### `bp-ptt-single` — Per-patient PTT vs BP

Visualizes PTT-BP relationship for individual patients across multiple CSV files.

```bash
# Single CSV
python main_analyze.py bp-ptt-single --csv ./mirror1_cleaned_patient_info.csv

# Multiple CSVs, minimum 5 measurements per patient
python main_analyze.py bp-ptt-single \
    --csv ./mirror1_info.csv ./mirror2_info.csv \
    --min_measurements 5
```

### `sample-stats` — Sample length report

Scans `mirror*_auto_cleaned` and `mirror*_auto_cleaned_sqi` directories, counting total CSV rows per directory group.

```bash
python main_analyze.py sample-stats
```

### `count-points` — Data point counter

Counts raw (`merged_log.csv`) and cleaned (`.csv` in `test_cleaned`) data points for a patient index range.

```bash
python main_analyze.py count-points
```

### `find-stim` — Stimulation patient finder

Scans all `patient_info.txt` files across mirror directories to identify patients with multiple measurement sessions (potential electrical stimulation).

```bash
python main_analyze.py find-stim
```

---

## `main_patient.py` — Patient Info Management

Two subcommands for working with `patient_info.txt` metadata.

| Subcommand | Script | Purpose |
|-----------|--------|---------|
| `extract` | `extract_patient_info.py` | Extract info from patient directories |
| `merge` | `merge_patient_info.py` | Merge extracted and manually-marked info CSVs |

### `extract` — Extract patient info

Reads `patient_info.txt` from patient directories and writes a CSV.

```bash
# Extract by mirror ID (uses auto-resolved data directory)
python main_patient.py extract --mirror-id mirror4

# Custom data directory
python main_patient.py extract \
    --data-dir ./custom_patient_data \
    --output ./patient_info.csv

# File mode with explicit data file
python main_patient.py extract \
    --mode file \
    --data-file ./patient_data/patient255/patient_info.txt \
    --output ./single_patient.csv
```

### `merge` — Merge patient info CSVs

Merges an automatically extracted CSV with a manually-marked CSV (e.g., with corrected BP values).

```bash
# Merge by mirror ID (auto-resolved file names)
python main_patient.py merge --mirror-id mirror4

# Lab data naming convention
python main_patient.py merge --lab --mirror-id lab_mirror1

# Explicit file paths
python main_patient.py merge \
    --extracted ./extracted_info.csv \
    --marked ./marked_info.csv \
    --output ./merged_info.csv
```

---

## Common Patterns

### Typical workflow

```bash
# 1. Slice raw recordings
python main_wash.py slice --input ./patient_data --output ./sliced

# 2. Auto-wash with fused SQI
python main_wash.py auto \
    --data_dir ./sliced \
    --output_dir ./cleaned \
    --sqi_method fused

# 3. Extract patient info
python main_patient.py extract --mirror-id mirror4

# 4. Visualize BP distribution
python main_analyze.py bp-dist --csv-dir . --bins 25

# 5. Run PTT-BP analysis
python main_analyze.py ptt-bp
```

### Getting help

```bash
# Entry point help
python main_wash.py --help
python main_analyze.py --help
python main_patient.py --help

# Subcommand help
python main_wash.py auto --help
python main_analyze.py bp-ptt --help
python main_patient.py merge --help
```
