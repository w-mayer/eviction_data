import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from rapidfuzz import process, fuzz

evictions_df = pd.read_csv('../DATA/cases_residential_only.txt')

llc_df = pd.read_csv('../DATA/LLC_clean.csv')

def preprocess_names(df, column_name):
    """
    Normalize names
    @df: pandas dataframe
    @column_name: str column name to preprocess
    
    """
    column = column_name + "_normalized"
    df[column] = (
        df[column_name]
        .str.lower()
        .str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    ) 
    return df


evictions_df = preprocess_names(evictions_df, 'plaintiff_name')
llc_df = preprocess_names(llc_df, 'Name')

# create flag 'is_llc' to restrict lookup to only llcs
llc_keywords = r"\b(llc|l\.l\.c|inc|inc\.|corporation|corp|corp\.|co|co\.|company|ltd|ltd\.|lp|l\.p\.|pllc|plc|plc\.|limited|limited liability company)\b"
evictions_df["is_llc"] = evictions_df["plaintiff_name_normalized"].str.contains(llc_keywords, regex=True)

# Now that we've set the 'is_llc' flag, we can remove these terms to reduce noise
suffixes = [
    "llc", "l.l.c", "inc", "incorporated", "corp", "corporation",
    "co", "company", "ltd", "l.p", "lp", "pllc", "plc", "llp", "p.c"
]

suffix_pattern = r'\s*(?:' + '|'.join(suffixes) + r')\.?\s*$'

evictions_df['plaintiff_name_normalized'] = evictions_df['plaintiff_name_normalized'].str.replace(suffix_pattern, '', regex=True)
llc_df['Name_normalized'] = llc_df['Name_normalized'].str.replace(suffix_pattern, '', regex=True)

evictions_df = evictions_df[evictions_df['is_llc']] # restrict lookup to only llcs
plaintiffs = evictions_df['plaintiff_name_normalized'].dropna().unique().tolist()
llcs = llc_df['Name_normalized'].dropna().unique().tolist()

# Function to process matching for a subset of plaintiff names
def process_chunk(plaintiff_chunk, llc_names, confidence=80):
    matches = {}
    for i, plaintiff in enumerate(plaintiff_chunk):
        try:
            best_match, score, _ = process.extractOne(plaintiff, llc_names, scorer=fuzz.WRatio)
            matches[plaintiff] = (best_match, score) if score >= confidence else (None, 0)
        except Exception as e:
            matches[plaintiff] = (None, 0)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} out of {len(plaintiff_chunk)} plaintiffs in this chunk.")
    return matches

# Function to split the plaintiff set and process in parallel
def parallel_match_plaintiffs(plaintiff_names_set, llc_names_set, chunk_size=1000):
    plaintiff_chunks = [list(plaintiff_names_set)[i:i + chunk_size] for i in range(0, len(plaintiff_names_set), chunk_size)]
    print(f"Total of {len(plaintiff_chunks)} chunks to process.")

    all_matches = {}
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, chunk, llc_names_set) for chunk in plaintiff_chunks]
        for i, future in enumerate(futures):
            all_matches.update(future.result())
            print(f"Chunk {i + 1} processed. Total matches found: {len(all_matches)}")
    
    return all_matches

chunk_size = len(plaintiffs) // 16 # adjust for num of cores, 16 worked well under an hour
# probably only needed ~50gb memory or less

# Matches looks like:
# {plaintiff_name: (best_match_llc, confidence_score)}
matches = parallel_match_plaintiffs(plaintiffs, llcs, chunk_size=chunk_size) 

evictions_df['match_tuple'] = evictions_df['plaintiff_name_normalized'].map(matches) # Match tuple looks like (best_match, confidence_score)
# Unpack into two separate columns
evictions_df[['best_match', 'match_confidence']] = pd.DataFrame(evictions_df['match_tuple'].tolist(), index=evictions_df.index)

evictions_df = pd.merge(
    left=evictions_df,
    right=llc_df,
    left_on='best_match',
    right_on='Name_normalized',
    how='left'
).drop(columns=['best_match', 'match_tuple'])

evictions_df.to_parquet('evictions_matched.parquet')

matches = evictions_df.copy()
matches = matches.sort_values(by='match_confidence', ascending=False)
matches = matches[['plaintiff_name', 'Name', 'match_confidence']]
matches = matches.drop_duplicates()
matches.to_csv('full_matches.csv')

