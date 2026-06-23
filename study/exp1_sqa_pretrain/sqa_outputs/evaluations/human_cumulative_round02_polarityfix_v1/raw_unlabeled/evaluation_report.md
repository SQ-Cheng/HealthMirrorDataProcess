# Unlabeled raw-ECG SQA evaluation

**Interpretation:** the raw dataset has no human SQA labels. Natural-data accuracy/AUROC cannot be identified. The results below measure deployment behavior and controlled-corruption sensitivity, not clinical validity.

Evaluated 7587 windows from 2592 patient files.

## Weak-label validation results

| model | best_epoch | val_loss | val_auroc_qrs | val_auroc_morph |
| --- | --- | --- | --- | --- |
| resnet_human_r02 | 81 | 0.2291 | 0.9691 | 0.9758 |
| resnet_weak | 17 | 0.3950 | 0.8869 | 0.8838 |
| tcn_human_r02 | 100 | 0.1842 | 0.9866 | 0.9839 |
| tcn_weak | 32 | 0.3831 | 0.9976 | 0.9972 |

## Natural raw-data summary

| model | mean_p_qrs | mean_p_morph | qrs_accept_0.8 | morph_accept_0.8 |
| --- | --- | --- | --- | --- |
| resnet_human_r02 | 0.4225 | 0.4304 | 0.2881 | 0.2943 |
| resnet_weak | 0.8505 | 0.8704 | 0.8148 | 0.8611 |
| tcn_human_r02 | 0.4959 | 0.4964 | 0.3706 | 0.3702 |
| tcn_weak | 0.8648 | 0.8859 | 0.7997 | 0.8549 |

## Severe-corruption mean probability change

| model | task | gaussian | baseline | impulse | clipping | dropout |
| --- | --- | --- | --- | --- | --- | --- |
| resnet_human_r02 | morph | -0.1355 | -0.0383 | -0.0104 | -0.1149 | -0.0631 |
| resnet_human_r02 | qrs | -0.1385 | -0.0790 | -0.0086 | -0.1170 | -0.0323 |
| resnet_weak | morph | 0.0043 | 0.0594 | 0.0101 | -0.0978 | -0.0831 |
| resnet_weak | qrs | 0.0148 | 0.0655 | 0.0154 | -0.1033 | -0.0776 |
| tcn_human_r02 | morph | -0.2773 | -0.1683 | -0.4705 | -0.1098 | 0.1347 |
| tcn_human_r02 | qrs | -0.2928 | -0.1648 | -0.4694 | -0.1068 | 0.1299 |
| tcn_weak | morph | 0.0017 | -0.0037 | -0.2184 | -0.0027 | -0.0028 |
| tcn_weak | qrs | -0.0027 | 0.0074 | -0.2280 | -0.0067 | -0.0038 |

## Fraction of samples whose probability decreased

| model | task | gaussian | baseline | impulse | clipping | dropout |
| --- | --- | --- | --- | --- | --- | --- |
| resnet_human_r02 | morph | 0.6484 | 0.4922 | 0.4727 | 0.5234 | 0.6758 |
| resnet_human_r02 | qrs | 0.6602 | 0.5391 | 0.4570 | 0.5273 | 0.6133 |
| resnet_weak | morph | 0.5742 | 0.1016 | 0.4336 | 0.9219 | 0.9297 |
| resnet_weak | qrs | 0.4922 | 0.0938 | 0.3750 | 0.9219 | 0.9141 |
| tcn_human_r02 | morph | 0.7383 | 0.7227 | 0.9922 | 0.5586 | 0.2344 |
| tcn_human_r02 | qrs | 0.7422 | 0.7266 | 0.9922 | 0.5586 | 0.2227 |
| tcn_weak | morph | 0.6445 | 0.7031 | 0.9727 | 0.6797 | 0.6953 |
| tcn_weak | qrs | 0.6484 | 0.6172 | 0.9766 | 0.6953 | 0.7070 |
