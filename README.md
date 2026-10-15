# Buyer and seller value from optional automated bargaining in electronic commerce

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23401250.svg)](https://doi.org/10.5281/zenodo.23401250)

Reproducibility materials for the manuscript of the same title by Bankatesh Choudhary and Meenakshi Sharma (Birla Institute of Technology Mesra), currently under review.

The study evaluates an optional seller-side bargaining agent against posted-price checkout. It compares two transparent rules and two frozen fitted Q iteration policies on matched synthetic opportunities across 193 evaluation products. The results are conditional computational evidence. They are not observations of real customers.

**Start with [`review_support/REVIEW_GUIDE.txt`](review_support/REVIEW_GUIDE.txt).** It maps every table and figure to its files, gives the replay commands, and documents formulas, coding and transformations.

## Contents

| Path | What it holds |
|---|---|
| `review_support/` | Review guide, variable definitions (`SOURCE_VARIABLES.txt`), source register with versions and checksums, file manifest, verification and reconstruction scripts |
| `multidataset_bargaining/` | Source processing (`prepare_data.py`), analyses, simulator, original AmazonHistoryPrice and CaSiNo files with their licence notices, cleaned tables |
| `bargaining_framework_revision/`, `bargaining_strengthening_20260926/` | Rule and learning experiments, frozen FQI checkpoints, history benchmark code and summaries |
| `focused_bargaining_20261002/` | Integrated checkout-relative channel evaluation, accounting tests, pinned `packaging/requirements.txt` |
| `hssc_submission_20261005/` | Robustness extension (seeds, category deletion, structural sensitivities), figure builder |
| `output/hssc_submission_20261005/Figures/` | Figures 1–6 (EPS, SVG, PNG) and figure data |

## Quick start

Use CPython 3.12 in a fresh virtual environment:

```bash
python -m venv .venv
.venv/bin/pip install -r focused_bargaining_20261002/packaging/requirements.txt   # Windows: .venv\Scripts\pip
python review_support/verify_files.py
```

The replay commands for Tables 4–6 and Figures 1–6 are listed in the review guide under **REPLAY COMMANDS**. Always write to new output directories.

Some `.joblib` files are pickles and can execute code when loaded. Verify their hashes first (`verify_files.py`), and inspect the code before loading them.

## CraigslistBargain inputs

The CraigslistBargain corpus (He et al., 2018) is **not included**. Its maintainers state no data licence, so we do not redistribute the raw JSON files or the 15 derived files listed in `SOURCE_OMISSIONS.json`. Instead, download the exact inputs from the maintainers' original CodaLab endpoints and rebuild everything locally:

```bash
python review_support/fetch_craigslist_sources.py --output craigslist_inputs --allow-unverified-tls
python review_support/reconstruct_sources.py --output reviewer_all_sources --craigslist-dir craigslist_inputs
python review_support/reconstruct_craigslist_derivatives.py --craigslist-dir craigslist_inputs --output reviewer_craigslist
```

The CodaLab TLS certificate expired on 7 October 2026. Integrity therefore rests on SHA-256: the fetch script keeps a file only if its checksum matches the study record, and stops at the certificate error unless you pass the flag.

On 15 October 2026 all three downloaded files matched the recorded checksums, and all 15 withheld files were regenerated:

- 8 were byte-identical.
- 4 gzip files were identical after decompression.
- 3 model checkpoints gave identical predictions.

The evidence is in `review_support/CRAIGSLIST_RECONSTRUCTION_CHECK.json`.

## Data sources and licences

| Source | Role | Licence | Included |
|---|---|---|---|
| AmazonHistoryPrice (Xia et al., 2024), [repository](https://github.com/TianXiaSJTU/AmazonPriceHistory) at revision `834ad90` | Product price anchors (930 products) | Apache-2.0 (`multidataset_bargaining/data/raw/amazon/LICENSE`) | Yes, original bytes |
| CaSiNo (Chawla et al., 2021), [repository](https://github.com/kushalchawla/CaSiNo) at revision `2f6ed4a` | Supporting outcome-measurement analysis | CC BY 4.0 (`multidataset_bargaining/data/raw/casino/LICENSE`) | Yes, original bytes |
| CraigslistBargain (He et al., 2018), [CodaLab worksheet](https://worksheets.codalab.org/worksheets/0x453913e76b65495d8b9730d41c7e0a0c) | Opening-offer inputs and history benchmark | Not stated by the maintainers | No; download script and checksums only |

Author code is released under the MIT licence ([`LICENSE`](LICENSE)). Author-generated results, simulations and figures are released under CC BY 4.0 ([`LICENSE-DATA.md`](LICENSE-DATA.md)). Third-party files keep their own licences. Nothing here grants rights in the CraigslistBargain corpus.

## Citation

Please cite the article once it is published, and cite the three source datasets:

- He H, Chen D, Balakrishnan A, Liang P (2018). *Decoupling strategy and generation in negotiation dialogues.* EMNLP 2018, 2333–2343. https://doi.org/10.18653/v1/D18-1256
- Xia T, et al. (2024). *Measuring bargaining abilities of LLMs: A benchmark and a buyer-enhancement method.* Findings of ACL 2024, 3579–3602. https://doi.org/10.18653/v1/2024.findings-acl.213
- Chawla K, et al. (2021). *CaSiNo: A corpus of campsite negotiation dialogues for automatic negotiation systems.* NAACL-HLT 2021, 3167–3185. https://doi.org/10.18653/v1/2021.naacl-main.254

Cite these materials as: Choudhary B, Sharma M (2026). *Buyer and seller value from optional automated bargaining in electronic commerce: reproducibility materials* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.23401250

That DOI identifies release v1.0.0. The concept DOI [10.5281/zenodo.23401249](https://doi.org/10.5281/zenodo.23401249) always resolves to the latest version. See also [`CITATION.cff`](CITATION.cff).
