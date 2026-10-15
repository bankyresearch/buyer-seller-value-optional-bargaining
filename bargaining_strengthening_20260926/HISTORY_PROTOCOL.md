# Matched-target history control

Specified locally on 2026-09-26 before fitting these models. This is an exploratory follow-up after inspection of the earlier corpus results, not a preregistered or independent external validation.

Question: does adding recognized message-price history improve prediction of the same native structured offers, holding examples and labels fixed?

Join the retained mixed-event and native-offer-only processed records by (dialogue, event). Restrict BOTH representations to identical native-offer targets. Require one-to-one matching and exact split, scenario, role, category, listing and target agreement. Preserve the original scenario-disjoint train/development/test partitions. Anticipated sample counts are 4,482/479/546 target events, with 371 test scenarios; any mismatch must be investigated before fitting.

Compare three information sets: mixed prior price history; native-offer prior price history; no-price-history context (listing, role, category and event position only). The last is a context baseline, not a model with literally no context. Neither historical view includes the current target. Use the existing 13 predictors for both history views and their four non-history predictors for the context view. Fit Ridge and absolute-loss histogram gradient boosting using the identical original tuning grid, training rows, random seed and development scenario-level MAE criterion. Train-only category encoding/scaling, target clipping to [0,2], no train-plus-development refit. Retain the training median reference. No hyperparameter choice may use test results.

Average absolute errors within dialogue, then within scenario. Report 2,000 paired scenario bootstrap percentile intervals for each information set and mixed-minus-native/context contrasts, separately for each model family. Report all families and contrasts without test-driven selection or multiplicity-adjusted significance claims. Negative contrasts mean mixed history reduces error. These are predictive-information controls, not semantic parser precision/recall, causal effects or evidence of deployment benefit.

Preserve all feature rows, selected models, development scores, test predictions, scenario errors, source hashes, and executable verification checks. Reproduction accepts an alternate output directory.
