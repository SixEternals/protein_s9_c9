# BL5-v4-PAM Shuffle Control Report

## 1. Executive Summary

This experiment tests whether BL5-v4-PAM's extra AUPRC over NoPAM depends on the correct PAM-to-sample correspondence.
The model architecture, formal split, labels, RNA-FM tokens, and LearnableRun features are kept fixed.
Only PAM features are shuffled within train/val/test separately using seed 42.
The formal test set is identical across models: 954326 samples, 3057 observed_positive, 951269 unobserved_candidate.
Real PAM reaches AUPRC=0.531281, while shuffled PAM reaches AUPRC=0.138883.
The PAM-minus-shuffle gap is 0.392398 AUPRC, supporting a real PAM correspondence signal.
The interpretation remains cautious because earlier PAM shortcut audits used the last three sequence characters, while this report uses canonical positions 21-23 to match the PAMEncoder.

## 2. Experimental Setup

- Split: `formal_split_bl5_seed42.json` (`sgrna_safe`).
- Base model: fine-tuned RNA-FM CLS + LearnableRunEncoder + PAM Encoder + simple concat classifier.
- Control: same model, but PAM features from positions 21-23 are shuffled within each split.
- Shuffle seed: 42 for train, 43 for val, 44 for test.
- Evaluation: explicit `best.pt` test evaluation with AUROC and AUPRC.

## 3. PAM Shuffle Audit

- train: samples=4697495, changed=4697494, unchanged=1, same_position_ratio=0.0
- val: samples=741552, changed=741552, unchanged=0, same_position_ratio=0.0
- test: samples=954326, changed=954324, unchanged=2, same_position_ratio=2e-06

Shuffle before/after PAM distributions are identical within each split; only sample correspondence changes.

## 4. Main Results

| model | PAM setting | test AUROC | test AUPRC | Accuracy | Precision | Recall | F1 | best_epoch | best_val_AUPRC | test_samples | test_observed_positive | test_unobserved_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BL0b-on-BL5split | baseline RNA-FM | 0.857756 | 0.295678 | 0.997379 | 0.782520 | 0.251881 | 0.381094 | 8 | N/A | 954326 | 3057 | 951269 |
| BL5-v4-NoPAM-control | no PAM encoder | 0.984098 | 0.502389 | 0.997327 | 0.652593 | 0.353942 | 0.458961 | 4 | 0.637471 | 954326 | 3057 | 951269 |
| BL5-v4-PAM | real PAM | 0.984194 | 0.531281 | 0.997606 | 0.768802 | 0.361138 | 0.491431 | 9 | 0.638364 | 954326 | 3057 | 951269 |
| BL5-v4-PAM-shuffle-control | within-split shuffled PAM | 0.669701 | 0.138883 | 0.997224 | 1.000000 | 0.133464 | 0.235498 | 6 | 0.232378 | 954326 | 3057 | 951269 |

## 5. Contribution Analysis

| contrast | AUPRC_delta | interpretation |
| --- | --- | --- |
| NoPAM - BL0b | 0.206711 | v4 no-PAM framework gain over pure RNA-FM |
| PAM - NoPAM | 0.028892 | approximate real PAM contribution in v4 |
| Shuffle - NoPAM | -0.363506 | effect of misleading shuffled PAM branch |
| PAM - Shuffle | 0.392398 | value of correct PAM correspondence |
| Shuffle - BL0b | -0.156795 | shuffle control vs pure RNA-FM baseline |

## 6. Stratified Analysis

### All
| model | subset | samples | observed_positive | unobserved_candidate | positive_ratio | AUROC | AUPRC | Accuracy | Precision | Recall | F1 | mean_prob_positive | median_prob_positive | mean_prob_unobserved_candidate | median_prob_unobserved_candidate | prob_gt_0_5_ratio_positive | prob_gt_0_5_ratio_unobserved_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BL0b-on-BL5split | All | 954326 | 3057 | 951269 | 0.003203 | 0.857756 | 0.295678 | 0.997379 | 0.782520 | 0.251881 | 0.381094 | 0.254487 | 0.000123 | 0.000358 | 0.000002 | 0.251881 | 0.000225 |
| BL5-v4-NoPAM-control | All | 954326 | 3057 | 951269 | 0.003203 | 0.984098 | 0.502389 | 0.997327 | 0.652593 | 0.353942 | 0.458961 | 0.496107 | 0.397139 | 0.035584 | 0.018525 | 0.353942 | 0.000606 |
| BL5-v4-PAM | All | 954326 | 3057 | 951269 | 0.003203 | 0.984194 | 0.531281 | 0.997606 | 0.768802 | 0.361138 | 0.491431 | 0.476243 | 0.363642 | 0.020421 | 0.007000 | 0.361138 | 0.000349 |
| BL5-v4-PAM-shuffle-control | All | 954326 | 3057 | 951269 | 0.003203 | 0.669701 | 0.138883 | 0.997224 | 1.000000 | 0.133464 | 0.235498 | 0.248937 | 0.134776 | 0.118237 | 0.119920 | 0.133464 | 0.000000 |

### NGG-only
| model | subset | samples | observed_positive | unobserved_candidate | positive_ratio | AUROC | AUPRC | Accuracy | Precision | Recall | F1 | mean_prob_positive | median_prob_positive | mean_prob_unobserved_candidate | median_prob_unobserved_candidate | prob_gt_0_5_ratio_positive | prob_gt_0_5_ratio_unobserved_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BL0b-on-BL5split | NGG-only | 819984 | 2349 | 817635 | 0.002865 | 0.792742 | 0.113649 | 0.997139 | 0.503513 | 0.091528 | 0.154899 | 0.094245 | 0.000040 | 0.000408 | 0.000003 | 0.091528 | 0.000259 |
| BL5-v4-NoPAM-control | NGG-only | 819984 | 2349 | 817635 | 0.002865 | 0.977830 | 0.324042 | 0.997054 | 0.468065 | 0.209025 | 0.288994 | 0.390064 | 0.352839 | 0.039848 | 0.022077 | 0.209025 | 0.000682 |
| BL5-v4-PAM | NGG-only | 819984 | 2349 | 817635 | 0.002865 | 0.977613 | 0.356243 | 0.997357 | 0.613750 | 0.209025 | 0.311845 | 0.352473 | 0.307148 | 0.022619 | 0.008916 | 0.209025 | 0.000378 |
| BL5-v4-PAM-shuffle-control | NGG-only | 819984 | 2349 | 817635 | 0.002865 | 0.619408 | 0.003810 | 0.997135 | 0.000000 | 0.000000 | 0.000000 | 0.135619 | 0.129121 | 0.122175 | 0.120624 | 0.000000 | 0.000000 |

### non-NGG-only
| model | subset | samples | observed_positive | unobserved_candidate | positive_ratio | AUROC | AUPRC | Accuracy | Precision | Recall | F1 | mean_prob_positive | median_prob_positive | mean_prob_unobserved_candidate | median_prob_unobserved_candidate | prob_gt_0_5_ratio_positive | prob_gt_0_5_ratio_unobserved_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BL0b-on-BL5split | non-NGG-only | 134342 | 708 | 133634 | 0.005270 | 0.995268 | 0.899732 | 0.998846 | 0.996409 | 0.783898 | 0.877470 | 0.786138 | 1.000000 | 0.000048 | 0.000000 | 0.783898 | 0.000015 |
| BL5-v4-NoPAM-control | non-NGG-only | 134342 | 708 | 133634 | 0.005270 | 0.999248 | 0.940733 | 0.998995 | 0.970443 | 0.834746 | 0.897494 | 0.847936 | 0.985962 | 0.009496 | 0.003050 | 0.834746 | 0.000135 |
| BL5-v4-PAM | non-NGG-only | 134342 | 708 | 133634 | 0.005270 | 0.999465 | 0.951397 | 0.999122 | 0.963836 | 0.865819 | 0.912202 | 0.886885 | 0.999997 | 0.006971 | 0.001635 | 0.865819 | 0.000172 |
| BL5-v4-PAM-shuffle-control | non-NGG-only | 134342 | 708 | 133634 | 0.005270 | 0.822465 | 0.604999 | 0.997767 | 1.000000 | 0.576271 | 0.731183 | 0.624901 | 1.000000 | 0.094142 | 0.099604 | 0.576271 | 0.000000 |

## 7. Paired Probability Analysis

# Paired Probability Comparison with Shuffle

| subset | samples | mean_delta_pam_minus_shuffle | median_delta_pam_minus_shuffle | prop_delta_pam_minus_shuffle_gt0 | prop_delta_pam_minus_shuffle_lt0 |
| --- | --- | --- | --- | --- | --- |
| All | 954326 | -0.096774 | -0.100085 | 0.029969 | 0.969945 |
| observed_positive | 3057 | 0.227306 | 0.152327 | 0.713772 | 0.259405 |
| unobserved_candidate | 951269 | -0.097816 | -0.100250 | 0.027771 | 0.972229 |
| NGG-only | 819984 | -0.098649 | -0.101386 | 0.031070 | 0.968930 |
| non-NGG-only | 134342 | -0.085331 | -0.097609 | 0.023247 | 0.976143 |

Positive `delta_pam_minus_shuffle` means the real PAM model assigned higher probability than the shuffled-PAM control for the same sample.

## 8. Interpretation

### 已经证明

- The formal split is consistent for BL0b, NoPAM, PAM, and PAM-shuffle after re-exporting NoPAM predictions.
- BL5-v4-PAM is stronger than BL0b on the same formal test set.
- BL5-v4-NoPAM-control is already a strong v4 no-PAM framework baseline.

### 本实验支持

- Correct PAM correspondence has substantial value: real PAM outperforms shuffled PAM by a large AUPRC margin.
- Shuffled PAM is not a harmless parameter-count control; it introduces misleading signal and performs below BL0b.

### 仍需谨慎

- PAM shortcut risk should be disclosed with an explicit PAM definition: canonical positions 21-23 versus `off_seq[-3:]` produce different stratified counts on variable-length CCLMoff sequences.
- Additional per-sgRNA, kNN, and in-silico perturbation analyses remain useful before broad biological claims.

## 9. Final Conclusion

BL5-v4-PAM-shuffle-control shows that breaking the PAM-sample correspondence drops AUPRC from 0.531281 with real PAM to 0.138883. This strongly supports that the PAM Encoder's gain depends on correct PAM information. Because CCLMoff contains variable-length off_seq values, PAM shortcut analyses must explicitly state whether PAM means canonical positions 21-23 or the last three sequence characters.
