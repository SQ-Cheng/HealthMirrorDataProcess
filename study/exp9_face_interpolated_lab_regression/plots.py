"""Interpolation-label audits and later same-target common-test comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_shape,target_grid_figsize
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS,TASK_UNITS
from study.exp2_face_pretrained_head32_regression.train import _regression_metrics

from . import config


def records(target):
    return pd.read_csv(config.OUTPUT/f"task_records/{target}.csv",dtype={"hospital_id":str,"video_id":str},float_precision="round_trip")


def panels():
    rows,columns=target_grid_shape(len(config.TARGETS))
    return plt.subplots(rows,columns,figsize=target_grid_figsize(rows,columns),squeeze=False)


def save(figure,figures,name):
    figure.tight_layout(rect=(0,.03,1,.94))
    for extension in ("png","pdf"):figure.savefig(figures/f"{name}.{extension}",dpi=180)
    plt.close(figure)


def plot_preparation():
    figures=config.OUTPUT/"figures";figures.mkdir(exist_ok=True)
    figure,axes=panels()
    for axis,target in zip(axes.flat,config.TARGETS):
        table=records(target).query("phase == 'post'")
        for variable,label,color in (("coverage_before_delta_h","Before video","#379A86"),("coverage_after_delta_h","After video","#CB6547")):
            values=np.sort(table[variable].to_numpy());axis.step(values,np.arange(1,len(values)+1)/len(values),where="post",label=label,color=color)
        axis.set(title=f"{TASK_LABELS[target]} | n={len(table)}",xlabel="Required bracketing-report distance (h)",ylabel="Cumulative fraction",xlim=(0,24),ylim=(0,1.02))
        axis.legend(fontsize=8);axis.grid(alpha=.15)
    figure.suptitle("Postoperative labels: strict same-phase bracketing within 24 hours on both sides")
    save(figure,figures,"interpolation_bracketing_distances")
    figure,axes=panels()
    for axis,target in zip(axes.flat,config.TARGETS):
        table=records(target).dropna(subset=["exp2_nearest_raw_value"]);x=table.exp2_nearest_raw_value.to_numpy();y=table.raw_value.to_numpy()
        axis.scatter(x,y,s=9,alpha=.4,color="#2878B5",edgecolors="none")
        low=min(x.min(),y.min());high=max(x.max(),y.max());axis.plot([low,high],[low,high],"--",color="#777777",linewidth=.8)
        if np.ptp(x):
            slope,intercept=np.polyfit(x,y,1);xx=np.array([x.min(),x.max()]);axis.plot(xx,slope*xx+intercept,color="#CB6547",linewidth=1)
        axis.set(title=TASK_LABELS[target],xlabel=f"Exp2 nearest value ({TASK_UNITS[target]})",ylabel=f"Exp9 mixed target ({TASK_UNITS[target]})")
        axis.grid(alpha=.15)
    figure.suptitle("Target change: Exp2 nearest-any-phase versus Exp9 preoperative-nearest/postoperative-interpolation")
    save(figure,figures,"interpolated_vs_nearest_targets")
    labs=pd.read_csv(config.SOURCE/"source_data/lab_timeseries.csv",dtype={"hospital_id":str})
    figure,axes=panels();example_rows=[]
    for example,(axis,target) in enumerate(zip(axes.flat,config.TARGETS),start=1):
        table=records(target)
        patient=table.groupby("hospital_id").size().idxmax()
        row=table.loc[table.hospital_id.eq(patient)].iloc[0]
        same=table.loc[table.hospital_id.eq(patient)&table.admission_unix.eq(row.admission_unix)&table.discharge_unix.eq(row.discharge_unix)]
        prefix=config.reference.SCORE_DEFINITIONS[target]["value_column"].removesuffix("_value")
        source=labs.loc[labs.hospital_id.eq(patient)&labs.analyte.eq(prefix)&labs.timestamp_unix.between(row.admission_unix,row.discharge_unix)].sort_values("timestamp_unix")
        origin=row.surgery_end_unix
        for phase,mask,color in (("Preoperative",source.timestamp_unix.lt(row.surgery_start_unix),"#379A86"),
                                 ("Postoperative",source.timestamp_unix.ge(origin),"#2878B5")):
            selected=source.loc[mask]
            if len(selected):axis.plot((selected.timestamp_unix-origin)/86400,selected.value,"o-",markersize=3,linewidth=1,color=color,label=phase)
        axis.axvspan((row.surgery_start_unix-origin)/86400,0,color="#E5E7EB",label="CABG interval")
        for phase,label,marker,color in (("pre","Pre-video observed target","s","#379A86"),("post","Post-video interpolated target","x","#CB6547")):
            targets=same.loc[same.phase.eq(phase)]
            axis.scatter((targets.label_time_unix-origin)/86400,targets.raw_value,s=24,color=color,marker=marker,label=label,zorder=3)
        axis.set(title=f"{TASK_LABELS[target]} | Example {example}",xlabel="Days relative to CABG end",ylabel=TASK_UNITS[target]);axis.grid(alpha=.15);axis.legend(fontsize=7)
        for video in same.itertuples():example_rows.append({"example":example,"target":target,"hospital_id":patient,"video_id":video.video_id,"phase":video.phase})
    figure.suptitle("Preoperative observed labels and postoperative interpolation; no surgery-crossing labels")
    save(figure,figures,"interpolation_curve_examples")
    pd.DataFrame(example_rows).to_csv(config.OUTPUT/"interpolation_example_records.csv",index=False)
    figure,axes=panels()
    for axis,target in zip(axes.flat,config.TARGETS):
        table=records(target).query("phase == 'pre'");values=np.sort(table.selected_lab_delta_h.to_numpy())
        if len(values):axis.step(values,np.arange(1,len(values)+1)/len(values),where="post",color="#379A86")
        axis.axvline(24,color="#CB6547",linestyle="--",linewidth=.8)
        axis.set(title=f"{TASK_LABELS[target]} | n={len(table)}",xlabel="Nearest preoperative report distance (h)",ylabel="Cumulative fraction",ylim=(0,1.02))
        axis.grid(alpha=.15)
    figure.suptitle("Preoperative targets have no maximum report/video distance; 24h shown only as reference")
    save(figure,figures,"preoperative_nearest_distances")


def plot_main_comparison(output,source):
    output,source=Path(output),Path(source);figures=output/"figures";figures.mkdir(exist_ok=True)
    if not (source/"COMPLETE").exists():
        (output/"comparison_status.json").write_text(json.dumps({"status":"pending","reason":"Exp2 main reference training/figures are not complete","command":"python -m study.exp9_face_interpolated_lab_regression.plots --compare"},indent=2)+"\n")
        return
    metrics=[];predictions=[]
    for target in config.TARGETS:
        original=pd.read_csv(source/f"runs/efficientnet_b0/{target}/video_predictions.csv",dtype={"hospital_id":str,"video_id":str},float_precision="round_trip")
        candidate=pd.read_csv(output/f"runs/efficientnet_b0/{target}/video_predictions.csv",dtype={"hospital_id":str,"video_id":str},float_precision="round_trip")
        shared=candidate.loc[candidate.split.eq("test")].merge(original.loc[original.split.eq("test"),["hospital_id","video_id","y_pred"]],on=["hospital_id","video_id"],suffixes=("_exp9","_exp2"),validate="one_to_one")
        if shared.empty:raise RuntimeError(f"No shared held-out test videos for {target}")
        shared.insert(0,"target",target);predictions.append(shared)
        for model in ("exp2","exp9"):
            values=_regression_metrics(shared.y_true,shared[f"y_pred_{model}"],shared.score_threshold,config.reference.SCORE_DEFINITIONS[target]["direction"])
            metrics.append({"target":target,"model":model,**values})
    table=pd.DataFrame(metrics);table.to_csv(output/"common_interpolated_test_metrics.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_csv(output/"common_interpolated_test_predictions.csv",index=False)
    for metric,title in (("mae","MAE"),("rmse","RMSE"),("pearson_r","Pearson r"),("r2","R2")):
        figure,axes=panels()
        for axis,target in zip(axes.flat,config.TARGETS):
            selected=table.loc[table.target.eq(target)].set_index("model").loc[["exp2","exp9"]]
            bars=axis.bar(range(2),selected[metric],color=("#87959E","#2878B5"));axis.bar_label(bars,fmt="%.3f",padding=3,fontsize=8)
            axis.set(title=f"{TASK_LABELS[target]} | n={int(selected.iloc[0]['n'])}",xticks=range(2),xticklabels=["Exp2 nearest-label","Exp9 mixed labels"],ylabel=title)
            axis.tick_params(axis="x",labelsize=8);axis.grid(axis="y",alpha=.15);axis.set_axisbelow(True)
        figure.suptitle(f"Both models scored against the same Exp9 target on common test videos | {title}")
        save(figure,figures,f"exp2_comparison_{metric}")
    (output/"comparison_status.json").write_text(json.dumps({"status":"complete","truth":"Exp9 observed-preoperative/interpolated-postoperative labels for both models","scope":"intersection of held-out videos; not a comparison of original unmatched main metrics"},indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--compare",action="store_true");args=parser.parse_args()
    if args.compare:plot_main_comparison(config.OUTPUT,config.SOURCE)
    else:plot_preparation()
