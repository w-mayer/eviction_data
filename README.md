## LLC Matching and Eviction Analysis Project

This project performs fuzzy matching of plaintiff names from eviction records to LLC names, with the goal of identifying the best match for each plaintiff name to an entity name in the LLC dataset. It leverages parallel processing and memory-efficient techniques to handle large datasets and ensures that the matching process can scale with a large volume of data.

## Project Structure
- **[`va_evictions.ipynb`](va_evictions.ipynb)**: The main script for processing the datasets, performing fuzzy matching, and merging the results.

  
## Requirements

- Python 3.x
- `pandas`
- `fuzzywuzzy` for fuzzy matching
- `concurrent.futures` for parallel processing
- `numpy`

You can install the required dependencies using `pip`:

```bash
pip install pandas fuzzywuzzy numpy
```
## License
This project is licensed under the MIT License.
