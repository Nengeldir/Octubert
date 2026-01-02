# Metrics Comparison

| Metric | model1 | model2 | model3 |
|--------|--------|--------|--------|
| bar_onset_density_mean | 0.5995 ± 0.0259 (95% CI [0.5477, 0.6488]) ✅ | 0.9927 ± 0.0004 (95% CI [0.9919, 0.9934]) | <span style='color:blue'><b>0.9987 ± 0.0001 (95% CI [0.9984, 0.9989])</b></span> |
| bar_pitch_var_mean | 29.1596 ± 2.6883 (95% CI [23.8072, 34.2559]) | 5381.2344 ± 161.4935 (95% CI [5060.8589, 5686.9404]) | <span style='color:blue'><b>324809.6562 ± 1162.8112 (95% CI [322620.7438, 327086.6969])</b></span> |
| consistency_duration | <span style='color:blue'><b>0.9363 ± 0.0154 (95% CI [0.9035, 0.9651]) ✅</b></span> | nan | nan |
| consistency_pitch | <span style='color:blue'><b>0.9457 ± 0.0155 (95% CI [0.9145, 0.9749]) ✅</b></span> | nan | nan |
| duration_js | <span style='color:blue'><b>0.0000 ± 0.0000 (95% CI [0.0000, 0.0000]) ✅</b></span> | 0.3701 ± 0.0198 (95% CI [0.3399, 0.4134]) | 0.0171 ± 0.0004 (95% CI [0.0166, 0.0183]) ✅ |
| harmonic_flux_gap_to_ref | <span style='color:blue'><b>-0.0507 ± 0.0067 (95% CI [-0.0638, -0.0381]) ✅</b></span> | -0.0085 ± 0.0056 (95% CI [-0.0197, 0.0024]) ✅ | 0.0537 ± 0.0055 (95% CI [0.0426, 0.0641]) |
| harmonic_flux_mean | 0.0943 ± 0.0046 (95% CI [0.0856, 0.1033]) | 0.1365 ± 0.0026 (95% CI [0.1317, 0.1416]) | <span style='color:blue'><b>0.1988 ± 0.0029 (95% CI [0.1935, 0.2045])</b></span> |
| key_agreement_with_ref | <span style='color:blue'><b>1.0000 ± 0.4809 (95% CI [0.0000, 1.0000]) ✅</b></span> | <span style='color:blue'><b>1.0000 ± 0.4975 (95% CI [0.0000, 1.0000]) ✅</b></span> | <span style='color:blue'><b>1.0000 ± 0.4975 (95% CI [0.0000, 1.0000]) ✅</b></span> |
| key_consistency | <span style='color:blue'><b>0.8088 ± 0.0230 (95% CI [0.7605, 0.8530]) ✅</b></span> | 0.1934 ± 0.0043 (95% CI [0.1853, 0.2017]) | 0.5503 ± 0.0083 (95% CI [0.5339, 0.5654]) ✅ |
| key_consistency_gap_to_ref | 0.4338 ± 0.0241 (95% CI [0.3838, 0.4790]) | <span style='color:blue'><b>-0.1816 ± 0.0116 (95% CI [-0.2043, -0.1575]) ✅</b></span> | 0.1753 ± 0.0137 (95% CI [0.1475, 0.2024]) |
| phrase_similarity_gap_to_ref | 0.1233 ± 0.0189 (95% CI [0.0830, 0.1560]) | -0.0969 ± 0.0105 (95% CI [-0.1175, -0.0773]) ✅ | <span style='color:blue'><b>-0.6286 ± 0.0139 (95% CI [-0.6551, -0.6017]) ✅</b></span> |
| phrase_similarity_mean | <span style='color:blue'><b>0.9225 ± 0.0192 (95% CI [0.8815, 0.9547])</b></span> | 0.7023 ± 0.0065 (95% CI [0.6896, 0.7146]) | 0.1706 ± 0.0103 (95% CI [0.1506, 0.1905]) |
| pitch_class_js | 0.0153 ± 0.0090 (95% CI [0.0103, 0.0449]) ✅ | <span style='color:blue'><b>0.0081 ± 0.0035 (95% CI [0.0052, 0.0188]) ✅</b></span> | 0.0082 ± 0.0037 (95% CI [0.0053, 0.0198]) ✅ |
| self_similarity_gap_to_ref | 0.2734 ± 0.0300 (95% CI [0.2079, 0.3275]) | -0.1770 ± 0.0170 (95% CI [-0.2088, -0.1415]) ✅ | <span style='color:blue'><b>-0.5112 ± 0.0148 (95% CI [-0.5396, -0.4814]) ✅</b></span> |
| self_similarity_mean | <span style='color:blue'><b>0.8093 ± 0.0244 (95% CI [0.7576, 0.8499])</b></span> | 0.3589 ± 0.0067 (95% CI [0.3461, 0.3727]) ✅ | 0.0247 ± 0.0023 (95% CI [0.0205, 0.0292]) |
| variance_duration | <span style='color:blue'><b>0.4044 ± 0.1624 (95% CI [0.1918, 0.8438])</b></span> | nan | nan |
| variance_pitch | <span style='color:blue'><b>0.0000 ± 0.0583 (95% CI [0.0000, 0.0000])</b></span> | nan | nan |
