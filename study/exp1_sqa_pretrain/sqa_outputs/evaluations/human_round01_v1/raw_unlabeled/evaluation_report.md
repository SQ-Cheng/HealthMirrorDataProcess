# Unlabeled raw-ECG SQA evaluation

**Interpretation:** the raw dataset has no human SQA labels. Natural-data accuracy/AUROC cannot be identified. The results below measure deployment behavior and controlled-corruption sensitivity, not clinical validity.

Evaluated 7587 windows from 2592 patient files.

## Weak-label validation results

| model | best_epoch | val_loss | val_auroc_qrs | val_auroc_morph |
| --- | --- | --- | --- | --- |
| resnet_window | 100 | 0.1444 | 1.0000 | 1.0000 |
| tcn_window | 99 | 0.3104 | 0.9615 | 0.9615 |

## Natural raw-data summary

| model | mean_p_qrs | mean_p_morph | qrs_accept_0.8 | morph_accept_0.8 |
| --- | --- | --- | --- | --- |
| resnet_window | 0.2930 | 0.3176 | 0.1062 | 0.1219 |
| tcn_window | 0.4252 | 0.4262 | 0.1597 | 0.1481 |

## Severe-corruption mean probability change

| model | task | gaussian | baseline | impulse | clipping | dropout |
| --- | --- | --- | --- | --- | --- | --- |
| resnet_window | morph | 0.1290 | 0.0603 | 0.0250 | 0.1267 | -0.1253 |
| resnet_window | qrs | 0.1024 | -0.0046 | 0.0395 | 0.1138 | -0.1199 |
| tcn_window | morph | 0.1878 | -0.1947 | -0.4347 | 0.2411 | 0.0092 |
| tcn_window | qrs | 0.1976 | -0.1871 | -0.4302 | 0.2354 | 0.0178 |

## Fraction of samples whose probability decreased

| model | task | gaussian | baseline | impulse | clipping | dropout |
| --- | --- | --- | --- | --- | --- | --- |
| resnet_window | morph | 0.1289 | 0.3398 | 0.3711 | 0.3164 | 0.7617 |
| resnet_window | qrs | 0.1602 | 0.3984 | 0.3203 | 0.2773 | 0.7461 |
| tcn_window | morph | 0.1016 | 0.7109 | 0.9961 | 0.1484 | 0.6055 |
| tcn_window | qrs | 0.1016 | 0.7109 | 0.9961 | 0.1953 | 0.5898 |
