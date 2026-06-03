# BL5-v4-PAM-shuffle-control Report

## Four-Model Summary

| model | PAM setting | test AUROC | test AUPRC | Accuracy | Precision | Recall | F1 | best_epoch | best_val_AUPRC | test_samples | test_observed_positive | test_unobserved_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BL0b-on-BL5split | baseline RNA-FM | 0.857756 | 0.295678 | 0.997379 | 0.782520 | 0.251881 | 0.381094 | 8 | N/A | 954326 | 3057 | 951269 |
| BL5-v4-NoPAM-control | no PAM encoder | 0.984098 | 0.502389 | 0.997327 | 0.652593 | 0.353942 | 0.458961 | 4 | 0.637471 | 954326 | 3057 | 951269 |
| BL5-v4-PAM | real PAM | 0.984194 | 0.531281 | 0.997606 | 0.768802 | 0.361138 | 0.491431 | 9 | 0.638364 | 954326 | 3057 | 951269 |
| BL5-v4-PAM-shuffle-control | within-split shuffled PAM | 0.669701 | 0.138883 | 0.997224 | 1.000000 | 0.133464 | 0.235498 | 6 | 0.232378 | 954326 | 3057 | 951269 |

## Key AUPRC Deltas

| contrast | AUPRC_delta | interpretation |
| --- | --- | --- |
| NoPAM - BL0b | 0.206711 | v4 no-PAM framework gain over pure RNA-FM |
| PAM - NoPAM | 0.028892 | approximate real PAM contribution in v4 |
| Shuffle - NoPAM | -0.363506 | effect of misleading shuffled PAM branch |
| PAM - Shuffle | 0.392398 | value of correct PAM correspondence |
| Shuffle - BL0b | -0.156795 | shuffle control vs pure RNA-FM baseline |

## Conclusion

BL5-v4-PAM-shuffle-control shows that breaking the PAM-sample correspondence drops AUPRC from 0.531281 with real PAM to 0.138883. This strongly supports that the PAM Encoder's gain depends on correct PAM information. Because CCLMoff contains variable-length off_seq values, PAM shortcut analyses must explicitly state whether PAM means canonical positions 21-23 or the last three sequence characters.

Full report: `results/bl5_v4_pam_shuffle_control/final_shuffle_control_report.md`
