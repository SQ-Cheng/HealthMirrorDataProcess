# Exp2 history-only Head32 classification

True binary counterpart of `exp2_history_only_head32_regression`. It uses the
same strictly prior, same-analyte, same-admission histories and the same
16-dimensional history encoder plus Head32. The output is a binary logit trained
with weighted BCE. Troponin I is replaced by total bilirubin >21 umol/L.
