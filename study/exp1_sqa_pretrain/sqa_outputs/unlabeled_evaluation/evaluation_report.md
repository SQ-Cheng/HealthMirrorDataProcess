# Unlabeled raw-ECG SQA evaluation

**Interpretation:** the raw dataset has no human SQA labels. Natural-data accuracy/AUROC cannot be identified. The results below measure deployment behavior and controlled-corruption sensitivity, not clinical validity.

Evaluated 7587 windows from 2592 patient files.

## Weak-label validation results

| model | best_epoch | val_loss | val_auroc_qrs | val_auroc_morph |
| --- | --- | --- | --- | --- |
| resnet_reference | 17 | 0.4654 | 0.9033 | 0.8069 |
| resnet_window | 17 | 0.3950 | 0.8869 | 0.8838 |
| tcn_reference | 41 | 0.4544 | 0.9959 | 0.9941 |
| tcn_window | 32 | 0.3831 | 0.9976 | 0.9972 |

## Natural raw-data summary

| model | mean_p_qrs | mean_p_morph | qrs_accept_0.8 | morph_accept_0.8 |
| --- | --- | --- | --- | --- |
| resnet_reference | 0.8145 | 0.6783 | 0.7867 | 0.0148 |
| resnet_window | 0.8017 | 0.8168 | 0.6980 | 0.7935 |
| tcn_reference | 0.7909 | 0.6831 | 0.4695 | 0.0386 |
| tcn_window | 0.7455 | 0.7814 | 0.2500 | 0.4866 |

## Severe-corruption mean probability change

| model | task | gaussian | baseline | impulse | clipping | dropout |
| --- | --- | --- | --- | --- | --- | --- |
| resnet_reference | morph | 0.0765 | 0.1106 | 0.0677 | -0.1349 | -0.0492 |
| resnet_reference | qrs | 0.0467 | 0.0843 | 0.0270 | -0.1203 | -0.0553 |
| resnet_window | morph | 0.0386 | 0.1001 | 0.0218 | -0.1050 | -0.0723 |
| resnet_window | qrs | 0.0482 | 0.1012 | 0.0264 | -0.1119 | -0.0664 |
| tcn_reference | morph | 0.0724 | 0.0485 | -0.1166 | -0.0483 | 0.0487 |
| tcn_reference | qrs | 0.0590 | 0.0528 | -0.2011 | 0.0308 | -0.0017 |
| tcn_window | morph | 0.0716 | 0.0424 | -0.2062 | 0.0320 | 0.0016 |
| tcn_window | qrs | 0.0747 | 0.0635 | -0.1903 | 0.0353 | 0.0059 |

## Fraction of samples whose probability decreased

| model | task | gaussian | baseline | impulse | clipping | dropout |
| --- | --- | --- | --- | --- | --- | --- |
| resnet_reference | morph | 0.0352 | 0.0195 | 0.0430 | 0.9805 | 0.8047 |
| resnet_reference | qrs | 0.0469 | 0.0625 | 0.2188 | 0.9336 | 0.8633 |
| resnet_window | morph | 0.1016 | 0.0039 | 0.2148 | 0.9414 | 0.8945 |
| resnet_window | qrs | 0.0664 | 0.0234 | 0.2305 | 0.9453 | 0.8945 |
| tcn_reference | morph | 0.0742 | 0.1953 | 0.8633 | 0.8516 | 0.1328 |
| tcn_reference | qrs | 0.0742 | 0.1289 | 0.9844 | 0.2227 | 0.5469 |
| tcn_window | morph | 0.0742 | 0.2734 | 0.9648 | 0.3359 | 0.5391 |
| tcn_window | qrs | 0.0703 | 0.1797 | 0.9766 | 0.2891 | 0.4570 |
