"""Descriptive charts of lost matches, with explicit comparison denominators."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_shape,target_grid_figsize
from study.exp2_face_pretrained_head32_regression.config import TARGETS
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS,TASK_UNITS


KEEP="retained_12h"
LOST="lost_12_to_24h"
COLORS=("#2878B5","#CB6547")
LABELS=("Retained: <=12h","Lost: >12h to 24h")
MILESTONES=("Admission","CABG","Discharge","Tie","CABG unavailable")
MILESTONE_COLORS=("#379A86","#2878B5","#CB6547","#B3A13A","#B7BEC5")
PHASES=("Pre-CABG","Intra-CABG","Postop 0-1d","Postop 1-3d","Postop 3-7d","Postop >7d","CABG unavailable")
PHASE_COLORS=("#6E8898","#9A4B69","#2878B5","#379A86","#CCAA45","#CB6547","#B7BEC5")


def panels():
    rows,columns=target_grid_shape(len(TARGETS))
    return plt.subplots(rows,columns,figsize=target_grid_figsize(rows,columns),squeeze=False)


def finish(figure,figures,name,footer=None):
    if footer:figure.text(.5,.01,footer,ha="center",fontsize=8)
    figure.tight_layout(rect=(0,.055,1,.94))
    for extension in ("png","pdf"):
        figure.savefig(figures/f"{name}.{extension}",dpi=200)
    plt.close(figure)


def plot_overview(summary,figures):
    table=summary.set_index("target").loc[list(TARGETS)]
    figure,axes=plt.subplots(1,2,figsize=(14,6.5))
    y=np.arange(len(TARGETS))
    for axis,retained,total,title in (
        (axes[0],"pairs12h","pairs24h","Matched video-analyte pairs"),
        (axes[1],"patients12h","patients24h","Patients with any match for this analyte"),
    ):
        keep=table[retained].to_numpy();loss=table[total].to_numpy()-keep
        axis.barh(y,keep,color=COLORS[0],label="Retained at 12h")
        axis.barh(y,loss,left=keep,color=COLORS[1],label="Lost entirely at 12h")
        for position,(a,b) in enumerate(zip(keep,loss)):
            axis.text(a+b+table[total].max()*.015,position,f"{b} lost ({b/(a+b):.1%})",va="center",fontsize=8)
        axis.set(yticks=y,yticklabels=[TASK_LABELS[t] for t in TARGETS],xlabel="Count",title=title,xlim=(0,table[total].max()*1.35))
        axis.invert_yaxis();axis.grid(axis="x",alpha=.15);axis.set_axisbelow(True)
    axes[0].legend(loc="lower right",fontsize=8)
    figure.suptitle("24h to 12h: loss of eligible matches without changing clinical data or split")
    finish(figure,figures,"attrition_overview")


def plot_normality(summary,events,figures):
    table=summary.set_index("target").loc[list(TARGETS)]
    figure,axes=plt.subplots(1,2,figsize=(14,6.5))
    y=np.arange(len(TARGETS));height=.35
    for axis,patient_weighted in zip(axes,(False,True)):
        for position,(prefix,label,color) in enumerate(zip(("retained","lost"),LABELS,COLORS)):
            column=f"{prefix}_patient_mean_abnormal_fraction" if patient_weighted else f"{prefix}_abnormal_fraction"
            value=table[column].to_numpy()*100
            bars=axis.barh(y+(position-.5)*height,value,height,color=color,label=label)
            if patient_weighted:
                low=table[f"{prefix}_patient_mean_abnormal_ci95_low"].to_numpy()*100
                high=table[f"{prefix}_patient_mean_abnormal_ci95_high"].to_numpy()*100
                good=np.isfinite(low)&np.isfinite(high)
                axis.errorbar(value[good],(y+(position-.5)*height)[good],xerr=np.maximum(np.stack([value[good]-low[good],high[good]-value[good]]),0),fmt="none",ecolor="#222222",capsize=2,linewidth=.8)
                for index,estimate in enumerate(value):
                    offset=max(estimate,high[index]) if np.isfinite(high[index]) else estimate
                    axis.text(offset+1.2,y[index]+(position-.5)*height,f"{estimate:.1f}%",va="center",fontsize=8)
            else:
                axis.bar_label(bars,labels=[f"{x:.1f}%" for x in value],padding=2,fontsize=8)
        axis.set(yticks=y,yticklabels=[TASK_LABELS[t] for t in TARGETS],xlabel="Abnormal fraction (%)",xlim=(0,112),
                 title="Each patient has equal weight" if patient_weighted else "Each matched pair has equal weight")
        axis.invert_yaxis();axis.grid(axis="x",alpha=.15);axis.set_axisbelow(True)
    axes[0].legend(loc="lower right",fontsize=8)
    figure.suptitle("Are the discarded values predominantly normal or abnormal?")
    finish(figure,figures,"normal_abnormal_comparison","Saved clinical labels, not model predictions. Patient-weighted error bars: 95% patient-bootstrap CI.")
    statuses=("retained_only","shared","lost_only");status_labels=("Retained-only assays","Assays shared by both subsets","Assays lost entirely")
    figure,axis=plt.subplots(figsize=(11,7));height=.24
    for position,(status,label,color) in enumerate(zip(statuses,status_labels,(COLORS[0],"#379A86",COLORS[1]))):
        selected=events.loc[events.event_status.eq(status)].groupby("target").binary_label.mean().reindex(TARGETS)*100
        axis.barh(y+(position-1)*height,selected,height,color=color,label=label)
    axis.set(yticks=y,yticklabels=[TASK_LABELS[t] for t in TARGETS],xlabel="Abnormal unique assay events (%)",xlim=(0,100))
    axis.invert_yaxis();axis.legend(fontsize=8);axis.grid(axis="x",alpha=.15);axis.set_axisbelow(True)
    figure.suptitle("Assay-level normality after collapsing repeated video matches")
    finish(figure,figures,"unique_assay_normality")


def plot_milestones(pairs,figures):
    for entity in ("lab","video"):
        for variable,categories,colors,title in (
            ("nearest_milestone",MILESTONES,MILESTONE_COLORS,"Nearest clinical milestone"),
            ("cabg_phase",PHASES,PHASE_COLORS,"Clinical phase relative to CABG"),
        ):
            figure,axes=panels()
            for axis,target in zip(axes.flat,TARGETS):
                group=pairs.loc[pairs.target.eq(target)];left=np.zeros(2)
                for category,color in zip(categories,colors):
                    values=[]
                    for retention in (KEEP,LOST):
                        selected=group.loc[group.retention.eq(retention)]
                        values.append(selected[f"{entity}_{variable}"].eq(category).mean()*100)
                    axis.barh(range(2),values,left=left,color=color,label=category)
                    left+=np.asarray(values)
                keep=group.retention.eq(KEEP).sum();lost=len(group)-keep
                axis.set(title=TASK_LABELS[target],yticks=range(2),yticklabels=[f"Retained\nn={keep}",f"Lost\nn={lost}"],xlim=(0,100),xlabel="Matched pairs (%)")
                axis.invert_yaxis()
            handles,labels=axes.flat[0].get_legend_handles_labels()
            figure.legend(handles,labels,loc="lower center",ncol=4,fontsize=8,bbox_to_anchor=(.5,.012))
            figure.suptitle(f"{title} | measured at {'lab-report' if entity=='lab' else 'video'} time")
            finish(figure,figures,f"{entity}_{variable}")


def plot_ecdf(pairs,figures,variable,name,title,xlabel,raw=False):
    figure,axes=panels()
    for axis,target in zip(axes.flat,TARGETS):
        for retention,label,color in zip((KEEP,LOST),LABELS,COLORS):
            values=np.sort(pairs.loc[pairs.target.eq(target)&pairs.retention.eq(retention),variable].dropna().to_numpy())
            if len(values):axis.step(values,np.arange(1,len(values)+1)/len(values),where="post",label=f"{label} (n={len(values)})",color=color,linewidth=1.5)
        axis.set(title=TASK_LABELS[target],xlabel=f"Raw value ({TASK_UNITS[target]})" if raw else xlabel,ylabel="Cumulative fraction",ylim=(0,1.02))
        if "stay_fraction" in variable:axis.set_xlim(0,1)
        if variable=="episode_median_assay_gap_h":axis.set_xscale("log")
        axis.grid(alpha=.15);axis.legend(fontsize=7)
    figure.suptitle(title)
    finish(figure,figures,name)


def plot_attrition_heatmap(table,figures,column,name,title):
    names=(sorted(table.mirror.unique(),key=lambda value:int(value.removeprefix("mirror")))
           if column=="mirror" else ["train","val","test"])
    rate=table.pivot(index="target",columns=column,values="lost_fraction").reindex(index=TARGETS,columns=names)
    counts=table.pivot(index="target",columns=column,values="pairs24h").reindex(index=TARGETS,columns=names)
    figure,axis=plt.subplots(figsize=(11,6.5))
    image=axis.imshow(rate.to_numpy()*100,vmin=0,vmax=100,cmap="viridis",aspect="auto")
    for row in range(len(TARGETS)):
        for column in range(len(names)):
            value=rate.iloc[row,column]
            text=f"{value:.1%}\nn={int(counts.iloc[row,column])}" if np.isfinite(value) else "No matches"
            axis.text(column,row,text,ha="center",va="center",fontsize=8,color="white" if not np.isfinite(value) or value<.55 else "#222222")
    axis.set(xticks=range(len(names)),xticklabels=names,yticks=range(len(TARGETS)),yticklabels=[TASK_LABELS[t] for t in TARGETS])
    figure.colorbar(image,ax=axis,label="Lost fraction of 24h matched pairs (%)")
    figure.suptitle(title+" | percentages use each cell's original 24h denominator")
    finish(figure,figures,name)


def plot_direction(pairs,figures):
    figure,axes=panels();categories=("Lab before video","Lab inside video","Lab after video")
    for axis,target in zip(axes.flat,TARGETS):
        group=pairs.loc[pairs.target.eq(target)];bottom=np.zeros(2)
        for category,color in zip(categories,("#379A86","#B7BEC5","#CB6547")):
            values=[group.loc[group.retention.eq(retention),"match_direction"].eq(category).mean()*100 for retention in (KEEP,LOST)]
            axis.bar(range(2),values,bottom=bottom,color=color,label=category.replace("Lab ","Report "));bottom+=values
        axis.set(title=TASK_LABELS[target],xticks=range(2),xticklabels=["Retained <=12h","Lost >12h"],ylabel="Matched pairs (%)",ylim=(0,100));axis.tick_params(axis="x",labelsize=8)
    handles,labels=axes.flat[0].get_legend_handles_labels();figure.legend(handles,labels,loc="lower center",ncol=3)
    figure.suptitle("Did the matched laboratory report precede or follow the video?")
    finish(figure,figures,"matching_direction")


def plot_all(output):
    figures=output/"figures";figures.mkdir(exist_ok=True)
    pairs=pd.read_csv(output/"matched_pairs.csv",dtype={"hospital_id":str,"video_id":str})
    events=pd.read_csv(output/"unique_lab_events.csv",dtype={"hospital_id":str})
    summary=pd.read_csv(output/"attrition_summary.csv");mirror=pd.read_csv(output/"mirror_attrition.csv")
    split=pd.read_csv(output/"split_attrition.csv")
    plot_overview(summary,figures);plot_normality(summary,events,figures);plot_milestones(pairs,figures)
    plot_attrition_heatmap(mirror,figures,"mirror","mirror_attrition","Mirror-specific attrition")
    plot_attrition_heatmap(split,figures,"split","split_attrition","Attrition within fixed train/validation/test assignments")
    plot_direction(pairs,figures)
    for variable,name,title,xlabel,raw in (
        ("raw_value","raw_value_distributions","Raw assay distributions: discarded versus retained matches","",True),
        ("lab_stay_fraction","lab_hospital_stay_position","Lab-report position in the hospitalization","Fraction of admission-to-discharge interval",False),
        ("video_stay_fraction","video_hospital_stay_position","Video position in the hospitalization","Fraction of admission-to-discharge interval",False),
        ("episode_median_assay_gap_h","assay_sampling_cadence","Within-admission assay cadence of the matched patients","Median consecutive same-analyte gap (h, log scale)",False),
    ):plot_ecdf(pairs,figures,variable,name,title,xlabel,raw)
