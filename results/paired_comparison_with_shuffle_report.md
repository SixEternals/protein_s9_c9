# Paired Probability Comparison with Shuffle

| subset | samples | mean_delta_pam_minus_shuffle | median_delta_pam_minus_shuffle | prop_delta_pam_minus_shuffle_gt0 | prop_delta_pam_minus_shuffle_lt0 |
| --- | --- | --- | --- | --- | --- |
| All | 954326 | -0.096774 | -0.100085 | 0.029969 | 0.969945 |
| observed_positive | 3057 | 0.227306 | 0.152327 | 0.713772 | 0.259405 |
| unobserved_candidate | 951269 | -0.097816 | -0.100250 | 0.027771 | 0.972229 |
| NGG-only | 819984 | -0.098649 | -0.101386 | 0.031070 | 0.968930 |
| non-NGG-only | 134342 | -0.085331 | -0.097609 | 0.023247 | 0.976143 |

Positive `delta_pam_minus_shuffle` means the real PAM model assigned higher probability than the shuffled-PAM control for the same sample.