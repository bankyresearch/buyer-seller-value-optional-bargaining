# Data and results licence

The MIT licence in `LICENSE` covers the authors' source code (`*.py`).

**Author-generated material** is licensed under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
This covers synthetic simulation trajectories, frozen policy checkpoints, result
tables, figure data, figures and documentation written by the authors.
Copyright (c) 2026 Bankatesh Choudhary and Meenakshi Sharma.

**Third-party material keeps its own licence**:

- AmazonHistoryPrice files under `multidataset_bargaining/data/raw/amazon/`:
  Apache License 2.0 (see the `LICENSE` file in that folder). Xia et al. (2024).
- CaSiNo files under `multidataset_bargaining/data/raw/casino/`: CC BY 4.0 (see
  the `LICENSE` file in that folder). Chawla et al. (2021).
- Cleaned tables derived from these sources (`amazon_products.csv`,
  `casino_*.csv`) are subject to the corresponding source licence as well.

**CraigslistBargain** (He et al., 2018) is not included. Its maintainers state no
data licence, and nothing in this repository grants any right in that corpus.
Some simulation inputs use numerical opening-offer ratios summarised from its
training split, as described in the manuscript. The scripts in `review_support/`
let users download the original files from the maintainers' CodaLab endpoints
and verify them by SHA-256.

No endorsement by the source authors is implied.
