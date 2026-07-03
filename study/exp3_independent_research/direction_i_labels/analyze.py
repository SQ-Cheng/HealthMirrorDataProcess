"""Direction I: Clinically Meaningful Label Redefinition.

Current labels are simple threshold-based (e.g., lactate > 2.0 mmol/L).
These are too coarse for a post-CABG population where:
  - Troponin ALWAYS rises after surgery (myocardial injury from CABG itself)
  - pO2 depends on ventilation/oxygen supplementation
  - Glucose is affected by stress, insulin, feeding
  - Lactate dynamics matter more than absolute threshold

Proposed redefined labels:
  1. lactate_clearance_failure: post-op lactate not decreasing as expected
  2. hb_drop_severity: relative Hb drop from pre-op baseline
  3. troponin_excessive: post-op troponin > 90th percentile for days-from-surgery
  4. oxygenation_delayed: pO2 not recovering to >80 within 3 days post-op
  5. multi_deterioration: ≥2 analytes worsening simultaneously
  6. delayed_recovery: any analyte not normalized by post-op day 3
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from ..config import OUTPUT_DIR, SEED

ANALYTES = ["lactate", "troponin", "glucose", "hemoglobin", "po2", "pco2"]


def run_label_redefinition(output_dir=None):
    """Analyze and propose clinically meaningful redefined labels."""
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "phase2", "direction_i")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──
    features_path = os.path.join(OUTPUT_DIR, "phase2", "direction_a",
                                  "ecg_features_corrected.csv")
    if not os.path.exists(features_path):
        print("ERROR: Corrected dataset not found.")
        return

    df = pd.read_csv(features_path)
    df_surg = df[df["days_from_surgery"].notna()].copy()
    print(f"Loaded {len(df_surg)} samples with surgery timing")

    # ═══════════════════════════════════════════════════════════════════
    # 1. Lactate clearance analysis
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("1. Lactate Clearance Failure Detection")
    print("=" * 70)

    # For patients with post-op lactate measurements, check if lactate is
    # decreasing over time. Expected: lactate should peak around surgery day
    # and decline thereafter.

    lactate_data = df_surg[["hospital_id", "days_from_surgery",
                             "lactate_value"]].dropna(subset=["lactate_value"])
    lactate_data = lactate_data.sort_values(["hospital_id", "days_from_surgery"])

    clearance_results = []
    for hid, group in lactate_data.groupby("hospital_id"):
        group = group.sort_values("days_from_surgery")
        post_op = group[group["days_from_surgery"] >= 0]
        if len(post_op) < 2:
            continue

        # Check trend: is lactate decreasing?
        x = post_op["days_from_surgery"].values
        y = post_op["lactate_value"].values

        # Simple linear trend
        if len(x) >= 2:
            from scipy.stats import linregress
            slope, _, _, _, _ = linregress(x, y)
            max_val = y.max()
            last_val = y[-1]

            clearance_results.append({
                "hospital_id": hid,
                "n_postop_measurements": len(post_op),
                "lactate_trend_slope": float(slope),
                "lactate_max": float(max_val),
                "lactate_last": float(last_val),
                "clearance_failure": int(slope > 0 and last_val > 2.0),
                "peak_then_decline": int(slope < -0.1),
            })

    clearance_df = pd.DataFrame(clearance_results)
    n_failure = clearance_df["clearance_failure"].sum() if len(clearance_df) > 0 else 0
    n_peak_decline = clearance_df["peak_then_decline"].sum() if len(clearance_df) > 0 else 0
    print(f"  Patients with ≥2 post-op lactate: {len(clearance_df)}")
    print(f"  Clearance failure (rising lactate): {n_failure} "
          f"({n_failure/max(len(clearance_df),1):.1%})")
    print(f"  Expected decline pattern: {n_peak_decline} "
          f"({n_peak_decline/max(len(clearance_df),1):.1%})")

    # ═══════════════════════════════════════════════════════════════════
    # 2. Hemoglobin drop severity
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("2. Hemoglobin Drop Severity")
    print("=" * 70)

    hb_data = df_surg[["hospital_id", "days_from_surgery",
                        "hemoglobin_value"]].dropna(subset=["hemoglobin_value"])
    hb_data = hb_data.sort_values(["hospital_id", "days_from_surgery"])

    hb_drops = []
    for hid, group in hb_data.groupby("hospital_id"):
        group = group.sort_values("days_from_surgery")
        # Pre-op baseline (closest to day -1)
        pre_op = group[group["days_from_surgery"] < 0]
        post_op = group[group["days_from_surgery"] >= 0]

        if len(pre_op) > 0 and len(post_op) > 0:
            baseline_hb = pre_op.iloc[-1]["hemoglobin_value"]
            post_op_hb_min = post_op["hemoglobin_value"].min()
            hb_drop = baseline_hb - post_op_hb_min
            hb_drop_pct = hb_drop / max(baseline_hb, 1) * 100

            hb_drops.append({
                "hospital_id": hid,
                "hb_baseline": float(baseline_hb),
                "hb_postop_min": float(post_op_hb_min),
                "hb_drop_abs": float(hb_drop),
                "hb_drop_pct": float(hb_drop_pct),
                "severe_drop": int(hb_drop > 30),  # drop > 30 g/L
            })

    hb_drop_df = pd.DataFrame(hb_drops)
    if len(hb_drop_df) > 0:
        print(f"  Patients with pre+post Hb: {len(hb_drop_df)}")
        print(f"  Mean Hb drop: {hb_drop_df['hb_drop_abs'].mean():.1f} g/L "
              f"({hb_drop_df['hb_drop_pct'].mean():.1f}%)")
        print(f"  Severe drop (>30 g/L): {hb_drop_df['severe_drop'].sum()} "
              f"({hb_drop_df['severe_drop'].mean():.1%})")

    # ═══════════════════════════════════════════════════════════════════
    # 3. Troponin excessive elevation
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("3. Troponin Excessive Elevation (relative to population)")
    print("=" * 70)

    tn_data = df_surg[["hospital_id", "days_from_surgery",
                        "troponin_value"]].dropna(subset=["troponin_value"])

    # For each days-from-surgery bin, compute 90th percentile
    tn_data["days_bin"] = pd.cut(tn_data["days_from_surgery"],
                                  bins=[-np.inf, -7, -3, -1, 0, 1, 3, 7, 14, np.inf],
                                  labels=["< -7", "-7 to -3", "-3 to -1", "-1 to 0",
                                          "0 to 1", "1 to 3", "3 to 7", "7 to 14", "> 14"])

    bin_stats = {}
    for bin_name, group in tn_data.groupby("days_bin", observed=False):
        if len(group) < 5:
            continue
        p50 = group["troponin_value"].quantile(0.50)
        p90 = group["troponin_value"].quantile(0.90)
        p95 = group["troponin_value"].quantile(0.95)
        bin_stats[bin_name] = {"n": len(group), "median": p50, "p90": p90, "p95": p95}
        print(f"  {str(bin_name):>12s}: n={len(group):3d}, "
              f"median={p50:.0f}, p90={p90:.0f}, p95={p95:.0f}")

    # ═══════════════════════════════════════════════════════════════════
    # 4. Multi-analyte deterioration detection
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("4. Multi-Analyte Deterioration")
    print("=" * 70)

    # Direction of "bad" for each analyte
    bad_direction = {
        "lactate": "high",
        "troponin": "high",
        "glucose": "high",
        "hemoglobin": "low",
        "po2": "low",
        "pco2": "either",  # both low and high are bad
    }

    # For each patient, count how many analytes are simultaneously worsening
    multi_det = []
    for hid, group in df_surg.groupby("hospital_id"):
        group = group.sort_values("capture_time_unix")
        for i in range(1, len(group)):
            row_prev = group.iloc[i - 1]
            row_curr = group.iloc[i]
            n_worsening = 0
            worsening_list = []

            for a in ANALYTES:
                val_prev = row_prev.get(f"{a}_value")
                val_curr = row_curr.get(f"{a}_value")
                if pd.isna(val_prev) or pd.isna(val_curr):
                    continue

                if bad_direction[a] == "high":
                    if val_curr > val_prev:
                        n_worsening += 1
                        worsening_list.append(a)
                elif bad_direction[a] == "low":
                    if val_curr < val_prev:
                        n_worsening += 1
                        worsening_list.append(a)

            if n_worsening >= 2:
                multi_det.append({
                    "hospital_id": hid,
                    "n_worsening": n_worsening,
                    "worsening_analytes": "+".join(worsening_list),
                    "days_from_surgery": row_curr.get("days_from_surgery", np.nan),
                })

    print(f"  Events with ≥2 analytes worsening: {len(multi_det)}")
    if len(multi_det) > 0:
        multi_df = pd.DataFrame(multi_det)
        post_op_events = multi_df[multi_df["days_from_surgery"] >= 0]
        print(f"  Post-operative multi-deterioration: {len(post_op_events)}")
        print(f"  Most common combinations:")
        combos = multi_df["worsening_analytes"].value_counts().head(5)
        for combo, count in combos.items():
            print(f"    {combo}: {count}")

    # ═══════════════════════════════════════════════════════════════════
    # 5. Proposal summary
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("5. Proposed Label Redefinitions")
    print("=" * 70)

    proposals = [
        {
            "label": "lactate_clearance_failure",
            "definition": "Post-op lactate slope > 0 AND last value > 2.0",
            "rationale": "Failure to clear lactate indicates ongoing hypoperfusion",
            "feasibility": f"~{n_failure} patients eligible",
        },
        {
            "label": "hb_drop_severe",
            "definition": "Hb drop > 30 g/L from pre-op baseline",
            "rationale": "Absolute drop matters more than crossing a threshold",
            "feasibility": f"~{hb_drop_df['severe_drop'].sum() if len(hb_drop_df)>0 else '?'} patients",
        },
        {
            "label": "troponin_excessive",
            "definition": "TnI > 90th percentile for days-from-surgery bin",
            "rationale": "Accounts for expected post-CABG elevation",
            "feasibility": "Per-bin p90 available from population",
        },
        {
            "label": "multi_deterioration",
            "definition": "≥2 analytes worsening simultaneously",
            "rationale": "Systemic deterioration vs isolated abnormality",
            "feasibility": f"~{len(multi_det)} events detected",
        },
        {
            "label": "delayed_recovery",
            "definition": "Any analyte not normalized by post-op day 3",
            "rationale": "Clinically meaningful composite endpoint",
            "feasibility": "Computable from trajectory data",
        },
    ]

    for p in proposals:
        print(f"\n  {p['label']}:")
        print(f"    Definition: {p['definition']}")
        print(f"    Rationale:  {p['rationale']}")
        print(f"    Feasibility: {p['feasibility']}")

    # Save
    pd.DataFrame(proposals).to_csv(os.path.join(output_dir, "proposed_labels.csv"), index=False)
    if len(clearance_df) > 0:
        clearance_df.to_csv(os.path.join(output_dir, "lactate_clearance.csv"), index=False)
    if len(hb_drop_df) > 0:
        hb_drop_df.to_csv(os.path.join(output_dir, "hb_drop_severity.csv"), index=False)

    print(f"\nResults saved to {output_dir}/")
    return proposals


if __name__ == "__main__":
    run_label_redefinition()
