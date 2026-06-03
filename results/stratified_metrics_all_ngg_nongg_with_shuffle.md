# Stratified Metrics by PAM Type with Shuffle

## All
| model | subset | samples | observed_positive | unobserved_candidate | positive_ratio | AUROC | AUPRC | Accuracy | Precision | Recall | F1 | mean_prob_positive | median_prob_positive | mean_prob_unobserved_candidate | median_prob_unobserved_candidate | prob_gt_0_5_ratio_positive | prob_gt_0_5_ratio_unobserved_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BL0b-on-BL5split | All | 954326 | 3057 | 951269 | 0.003203 | 0.857756 | 0.295678 | 0.997379 | 0.782520 | 0.251881 | 0.381094 | 0.254487 | 0.000123 | 0.000358 | 0.000002 | 0.251881 | 0.000225 |
| BL5-v4-NoPAM-control | All | 954326 | 3057 | 951269 | 0.003203 | 0.984098 | 0.502389 | 0.997327 | 0.652593 | 0.353942 | 0.458961 | 0.496107 | 0.397139 | 0.035584 | 0.018525 | 0.353942 | 0.000606 |
| BL5-v4-PAM | All | 954326 | 3057 | 951269 | 0.003203 | 0.984194 | 0.531281 | 0.997606 | 0.768802 | 0.361138 | 0.491431 | 0.476243 | 0.363642 | 0.020421 | 0.007000 | 0.361138 | 0.000349 |
| BL5-v4-PAM-shuffle-control | All | 954326 | 3057 | 951269 | 0.003203 | 0.669701 | 0.138883 | 0.997224 | 1.000000 | 0.133464 | 0.235498 | 0.248937 | 0.134776 | 0.118237 | 0.119920 | 0.133464 | 0.000000 |

## NGG-only
| model | subset | samples | observed_positive | unobserved_candidate | positive_ratio | AUROC | AUPRC | Accuracy | Precision | Recall | F1 | mean_prob_positive | median_prob_positive | mean_prob_unobserved_candidate | median_prob_unobserved_candidate | prob_gt_0_5_ratio_positive | prob_gt_0_5_ratio_unobserved_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BL0b-on-BL5split | NGG-only | 819984 | 2349 | 817635 | 0.002865 | 0.792742 | 0.113649 | 0.997139 | 0.503513 | 0.091528 | 0.154899 | 0.094245 | 0.000040 | 0.000408 | 0.000003 | 0.091528 | 0.000259 |
| BL5-v4-NoPAM-control | NGG-only | 819984 | 2349 | 817635 | 0.002865 | 0.977830 | 0.324042 | 0.997054 | 0.468065 | 0.209025 | 0.288994 | 0.390064 | 0.352839 | 0.039848 | 0.022077 | 0.209025 | 0.000682 |
| BL5-v4-PAM | NGG-only | 819984 | 2349 | 817635 | 0.002865 | 0.977613 | 0.356243 | 0.997357 | 0.613750 | 0.209025 | 0.311845 | 0.352473 | 0.307148 | 0.022619 | 0.008916 | 0.209025 | 0.000378 |
| BL5-v4-PAM-shuffle-control | NGG-only | 819984 | 2349 | 817635 | 0.002865 | 0.619408 | 0.003810 | 0.997135 | 0.000000 | 0.000000 | 0.000000 | 0.135619 | 0.129121 | 0.122175 | 0.120624 | 0.000000 | 0.000000 |

## non-NGG-only
| model | subset | samples | observed_positive | unobserved_candidate | positive_ratio | AUROC | AUPRC | Accuracy | Precision | Recall | F1 | mean_prob_positive | median_prob_positive | mean_prob_unobserved_candidate | median_prob_unobserved_candidate | prob_gt_0_5_ratio_positive | prob_gt_0_5_ratio_unobserved_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BL0b-on-BL5split | non-NGG-only | 134342 | 708 | 133634 | 0.005270 | 0.995268 | 0.899732 | 0.998846 | 0.996409 | 0.783898 | 0.877470 | 0.786138 | 1.000000 | 0.000048 | 0.000000 | 0.783898 | 0.000015 |
| BL5-v4-NoPAM-control | non-NGG-only | 134342 | 708 | 133634 | 0.005270 | 0.999248 | 0.940733 | 0.998995 | 0.970443 | 0.834746 | 0.897494 | 0.847936 | 0.985962 | 0.009496 | 0.003050 | 0.834746 | 0.000135 |
| BL5-v4-PAM | non-NGG-only | 134342 | 708 | 133634 | 0.005270 | 0.999465 | 0.951397 | 0.999122 | 0.963836 | 0.865819 | 0.912202 | 0.886885 | 0.999997 | 0.006971 | 0.001635 | 0.865819 | 0.000172 |
| BL5-v4-PAM-shuffle-control | non-NGG-only | 134342 | 708 | 133634 | 0.005270 | 0.822465 | 0.604999 | 0.997767 | 1.000000 | 0.576271 | 0.731183 | 0.624901 | 1.000000 | 0.094142 | 0.099604 | 0.576271 | 0.000000 |
