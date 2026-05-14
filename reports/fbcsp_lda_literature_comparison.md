# FBCSP/CSP + LDA Literature Comparison

This note compares MindPlay's FBCSP+LDA results with close classical motor-imagery BCI benchmarks. The fairest MindPlay values are the 5-fold cross-validation results, because saved-model accuracy may be optimistic when evaluated on calibration/session data.

## MindPlay Results by Trial Count

| Trial count | Dataset(s) | Fair FBCSP+LDA accuracy to report | Interpretation |
|---:|---|---:|---|
| 20 | S12_U, S14_U | 70.0% to 75.0% CV | Promising but low evidence; only 4 test trials per fold in 5-fold CV. |
| 40 | S15, S20, S21 | 75.0% to 87.5% CV | Stronger than the 20-trial results, but still small; one fold error changes accuracy a lot. |
| 50 | S11 | 48.0% CV | Near chance; not a good subject-level result. |
| 160 | S02 | 61.9% CV | More stable trial count, but modest accuracy. |
| 160 | raw NPZ models | 82.5% to 94.4% saved-model accuracy | High saved-model performance, but should not be compared as held-out/generalization accuracy unless CV or held-out testing is added. |

Summary across available subject CV datasets:

| Summary | Value |
|---|---:|
| Unweighted CV mean across S02, S11, S12_U, S14_U, S15, S20, S21 | 71.4% |
| Trial-weighted CV mean across those datasets | 67.6% |
| Unweighted CV mean excluding S11, which has no saved model | 75.3% |
| Trial-weighted CV mean excluding S11 | 70.6% |

## Closest Literature Benchmarks

Most public benchmark papers report Cohen's kappa rather than raw accuracy. For balanced datasets, approximate accuracy can be estimated as:

- Binary task: accuracy = 0.5 + 0.5 * kappa
- Four-class task: accuracy = 0.25 + 0.75 * kappa

| Source / benchmark | Method closest to MindPlay | Classes | Trial scale | Reported score | Approx. accuracy | Comparison with MindPlay |
|---|---|---:|---|---:|---:|---|
| BCI Competition IV, Dataset 2a, Kai Keng Ang et al. | FBCSP + Naive Bayes Parzen Window | 4 | 288 evaluation trials per subject | kappa = 0.57 avg | ~67.8% | Similar to MindPlay's trial-weighted subject CV mean of 67.6%, but Dataset 2a is harder because it is 4-class. |
| BCI Competition IV, Dataset 2a, Liu Guangquan et al. | CSP/log-variance + Fisher LDA reduction + Bayesian classifier | 4 | 288 evaluation trials per subject | kappa = 0.52 avg | ~64.0% | MindPlay's overall subject CV mean is slightly above this, but MindPlay is binary and uses fewer channels/trials. |
| BCI Competition IV, Dataset 2b, Zheng Yang Chin et al. | FBCSP + Naive Bayes Parzen Window | 2 | multi-session binary MI benchmark | kappa = 0.60 avg | ~80.0% | MindPlay's best 40-trial subjects, S15 and S21, are comparable or above this; overall CV average is below it. |
| BCI Competition IV, Dataset 2b, Huang Gan et al. | CSSD/CSP-style spatial features + LDA | 2 | multi-session binary MI benchmark | kappa = 0.58 avg | ~79.0% | This is the closest binary LDA-style benchmark. MindPlay's S15/S21 are strong, but the trial-weighted CV mean is lower. |

## Where MindPlay Lies

MindPlay's FBCSP+LDA performance is best described as follows:

- The strongest subject-level CV results, S15 at 82.5% and S21 at 87.5%, are in the good range for classical binary motor-imagery BCI and compare favorably with binary CSP/FBCSP benchmark accuracies around 79% to 80%.
- The overall trial-weighted CV mean of 67.6% is mid-range and should not be presented as state-of-the-art. It is close to the converted BCI Competition IV Dataset 2a FBCSP benchmark, but Dataset 2a is a harder four-class task.
- The 20-trial 100% saved-model scores should not be used as the main comparison because the trial count is too small and the evaluation may be calibration-set performance.
- The 160-trial S02 CV result of 61.9% is modest and shows that performance is subject-dependent.
- The raw NPZ saved-model result of 94.4% on 160 trials is high, but it needs held-out or CV validation before it can be compared with published benchmark accuracy.

Recommended thesis/paper wording:

> Compared with classical CSP/FBCSP motor-imagery benchmarks, the proposed FBCSP+LDA pipeline achieved subject-dependent performance. Five-fold CV accuracy ranged from 48.0% to 87.5%, with the strongest 40-trial subject datasets reaching 82.5% to 87.5%. These values are comparable to reported binary CSP/FBCSP benchmark accuracies around 79% to 80% for the best-performing subjects, although the overall trial-weighted CV mean of 67.6% indicates mid-range generalization performance. Therefore, the results support feasibility rather than a state-of-the-art performance claim.

Sources used:

- BCI Competition IV final results: https://www.bbci.de/competition/iv/results/index.html
- Dataset 2a result page reports FBCSP winner kappa = 0.57 and CSP/LDA-style entry kappa = 0.52.
- Dataset 2b result page reports FBCSP winner kappa = 0.60 and LDA-style entry kappa = 0.58.
