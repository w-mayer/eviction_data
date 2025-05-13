## LLC Matching and Eviction Analysis Project
# Last Update: 5/13/25

This project focuses on matching Virginia eviction records to registered LLC entities using fuzzy string matching and parallel processing techniques. It also includes exploratory data analysis (EDA) to uncover patterns in eviction filings.

## Project Structure
eviction_matching.ipynb runs the matching

confidence_testing.ipynb shows basic stats on the matching process

evictions_eda.ipynb shows some approaches for grouping entities for further work


- **[`SCRIPTS/eviction_matching.ipynb`](SCRIPTS/eviction_matching.ipynb)**: The primary notebook that processes eviction records and performs fuzzy matching using rapidfuzz.
- **[`SCRIPTS/confidence_testing.ipynb`](SCRIPTS/confidence_testing.ipynb)**: Script for resampling different confidence thresholds for finetuning.
- **[`SCRIPTS/evictions_eda.ipynb`](SCRIPTS/evictions_eda.ipynb)**: An exploratory notebook that applies simple heuristics to group and analyze eviction data, highlighting preliminary trends.

## Installation
- Python 3.x
- `pandas`
- `rapidfuzz` for fuzzy matching
- `concurrent.futures` for parallel processing

You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Notes
- The matching process uses rapidfuzz for efficient fuzzy matching
- The EDA notebook applies basic grouping heuristics; future analyses planned

## License
This project is licensed under the MIT License.
