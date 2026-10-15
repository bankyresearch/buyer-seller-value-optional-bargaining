FOCUSED OPTIONAL-BARGAINING REPRODUCTION PACKAGE
October 2026 exploratory revision

SCOPE
This archive replays four frozen seller-policy families in matched optional-
purchase-channel scenarios, then reproduces the accounting, comparisons and
threshold analyses. All 193 evaluation products, 80 opportunities per product,
five designated learner seeds, failures and all declared scenario/acceptance
cells are retained. Policy seeds are not independent customers or a deployed
ensemble. Prior test outcomes had been inspected. The study is exploratory,
not externally preregistered and not a field or human-agent experiment.

The archive preserves repository-relative sibling directories. Do not flatten
the ZIP. Start commands from its extracted top-level directory. The supplied
canonical results remain under focused_bargaining_20261002/*/results. Replay
outputs go to a newly created directory; existing results are never replaced by
the replay helper. ZIP binaries are not supplied as Python dependencies.

ENVIRONMENT
Tested CPython 3.12.14. Use a new virtual environment, then install:
python -m pip install -r focused_bargaining_20261002/packaging/requirements.txt

Dependencies must first be installed from trusted packages, or from a local
wheelhouse using pip --no-index --find-links YOUR_WHEELHOUSE. An offline run
requires those dependencies already installed. No API key, paid service or
network access is used by the replay. Eight package versions are pinned; binaries
and a Python interpreter are not vendored. Other operating
systems or library versions may change floating-point behavior. Never silently
replace unavailable package versions and claim byte-identical reproduction.

VERIFY FIRST
python focused_bargaining_20261002/packaging/verify_manifest.py

Frozen .joblib files use pickle and can execute code on loading. Verify the
archive's provenance and hashes; do not load untrusted replacements. The
manifest checks archive integrity, not the identity or trustworthiness of its
publisher. It excludes itself because a self-hash cannot be embedded unchanged.

FULL FROZEN-POLICY REPLAY
python focused_bargaining_20261002/packaging/replay.py --output replay_check

This runs unit checks, regenerates all 2,223,360 potential trajectories, checks
the grid, compares every trajectory numerically with canonical outputs, reruns
the channel analysis, and checks every current analysis table plus continuity
with the original rule-policy accounting. A fresh output directory is required.
Same-environment numerical tolerances are 1e-12; monetary cents checks allow
1e-5 in units of cents for floating-point representation. Wall-clock times,
absolute directory names and gzip metadata are not scientific outcomes.

Approximate observed compute: trajectory generation 2-4 minutes and channel
analysis under one minute on the original machine, plus tests/comparisons and
file I/O. Reviewer hardware varies. Reserve at least 1 GB free disk and several
GB memory. Two native CPU threads are configured by replay.py. This is a local
same-environment reproducibility check, not independent external validation.

ANALYSIS ONLY
python focused_bargaining_20261002/analysis/channel_analysis.py --inputs focused_bargaining_20261002/integration/results --output analysis_check
python focused_bargaining_20261002/analysis/test_channel_analysis.py --output analysis_check
python focused_bargaining_20261002/analysis/verify_analysis.py --current focused_bargaining_20261002/analysis/results --replay analysis_check

The verification program writes its report under --current. Run in a disposable
copy if every supplied byte must remain unchanged. The replay helper avoids
this by using a newly generated current result directory. Reports may include
canonical input hashes; regenerated gzip headers yield different hashes even
when numerical tables are identical. The replay helper uses canonical inputs
for analysis verification after separately verifying regenerated trajectories.

PRIOR SUPPORTING ANALYSES (NOT NEW EXTERNAL VALIDATION)
The September history-control source, six selected models, numerical inputs,
predictions, selections and summaries are retained. Its three history views
use identical native-offer targets. To rerun model fitting on those inputs:
python bargaining_strengthening_20260926/history_controls.py --output history_check

CaSiNo numerical participant/dialogue inputs and the original association
implementation/results are retained. To rerun only that association analysis:
python focused_bargaining_20261002/packaging/replay.py --output casino_check --casino-only

Reconstructing all derived tables from original dialogue events and verifying
their temporal extraction requires upstream raw Craigslist files, deliberately
omitted here. Use original-source URLs and SHA256 hashes in DATA_AND_RIGHTS.txt
and craigslist_audit.json. history_verify.py additionally requires --raw with
the original train.json, dev.json and test.json. Raw utterances and unused
personal attributes are not redistributed. Source reconstruction code is
preserved for audit; do not run its entry points expecting omitted raw files.

Training source is supplied for transparency. The present replay intentionally
loads ten frozen leaf-40 checkpoints and does not retrain or tune policies.
Original sixty-fit selection scores and its local protocol are included as
provenance, not as a newly executed analysis. A bounded separate retraining of
the two designated families can be run with learning_controls.py --stage train
--specs mixed_enriched_128 clock_mixed_enriched_128 --out-dir learner_check.
Do not run the legacy --stage all pipeline from this focused archive: older
imitation/comparison checkpoints not used by this paper are intentionally absent.

INTERPRETATION
Surpluses and costs are normalized by listing price per original opportunity.
Historical prices are proxy floors, not actual merchant costs. Numerical
failure price is missing, not zero. Failures can still incur attempted-round
cost. Participation, fallback and acceptance are assumed, not estimated demand.
The 95% product bootstrap intervals condition on these assumptions and the five
frozen seeds; they do not establish behavioral realism, causal field effects,
individual Pareto improvement or acceptance by any publication venue.

FIGURES
The final Fig1-Fig4 PNG/EPS/SVG assets, build_figures.py, numeric_plot_data.csv
and figure_verification.json are included. To rebuild into a separate directory:
python focused_bargaining_20261002/figures/build_figures.py --output figure_check
The builder may also generate convenience aliases and preview PNGs; these
redundant derivatives are omitted from this ZIP. Its historical verification
records can mention preview files that are recreated by the command. DejaVu
Sans is used; Matplotlib and Pillow are included in requirements. Set
MPLCONFIGDIR to a writable local cache directory when platform permissions
restrict the default cache. Main manuscripts and author documents are separate.

File-level provenance is in SHA256SUMS.json. Explicit inclusion rules are in
package_reproduction.py. No private author documents, raw dialogue utterances,
external-service credentials, duplicate replay directories, or vendored
dependencies are included. The main author-delivery archive may add manuscript
documents separately. Those files are not silently covered by this computational
archive's manifest.
