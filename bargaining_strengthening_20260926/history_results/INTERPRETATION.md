# Matched-target information control

All variants use the same 4,482 training, 479 development and 546 test native-offer targets. The test set contains 371 scenarios. Models are tuned on development performance only. Uncertainty is a 2,000-resample paired scenario bootstrap after averaging within dialogue and then scenario.

| Model family | Mixed-history MAE | Native-only-history MAE | Context-only MAE | Mixed minus native (95% interval) |
|---|---:|---:|---:|---:|
| Ridge | 0.07637 | 0.10317 | 0.10425 | −0.02680 [−0.03232, −0.02117] |
| Gradient boosting | 0.05297 | 0.10399 | 0.10399 | −0.05101 [−0.05903, −0.04256] |

Errors are normalized by listing price. The training-median reference MAE is 0.10357. The gradient-boosting improvement of 0.05101 is a decrease of about 5.10 percentage points of listing price in this error measure; it is not a 5.10% change in revenue, bargaining success, accuracy or seller utility.

The controlled comparison supports the predictive usefulness of previously recognized message-price observations for predicting native structured offers in the retained Craigslist corpus. It resolves the original ambiguity caused by simultaneously changing target rows and history representation. It does not measure semantic parser accuracy, independent external transfer or the economic value of the prediction module. Current targets were not included in the feature histories: all 5,507 native targets and both history views were independently reconstructed in event order from the original source records by the same computational team.

All model families, hyperparameter candidates, target predictions and scenario-level errors are retained. The contextual baseline retains listing, role, category and event position, so it should be described as **no-price-history context**, not a model with no information.
