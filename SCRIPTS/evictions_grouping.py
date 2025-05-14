import argparse
import pandas as pd
import numpy as np

def normalize_entity(name):
    """Normalize a field by stripping whitespace, converting to lowercase, and removing punctuation."""
    if pd.isna(name):
        return ''
    return name.strip().lower().replace('.', '').replace(',', '')

def normalize_columns(df, columns):
    """Normalize specified columns in the DataFrame."""
    for col in columns:
        df[f'{col}_normalized'] = df[col].apply(normalize_entity)
    return df

def create_composite_key(df, columns, key_name='Composite_Key'):
    """Create a composite key from specified columns."""
    df[key_name] = df[columns].fillna('').agg('|'.join, axis=1)
    return df, key_name

def analyze_llc_networks(df, composite_key):
    """Analyze networks based on the composite key."""
    if composite_key not in df.columns:
        raise ValueError(f"Composite key column '{composite_key}' not found in DataFrame")

    df = df.set_index(composite_key)

    # Group by the composite key
    groups = df.groupby(level=0).agg({
        'Name': 'count',
        'RA-Name': list,
        'plaintiff_name': list,
        'plaintiff_attorney': list,
        'IncorpDate': list,
        'Zip': list,
        'filed_date': list,
        'serial_filing': list
    }).rename(columns={'Name': 'LLC_Count'}).reset_index()

    # Filter for groups with more than one LLC
    groups = groups.loc[groups['LLC_Count'] > 1]

    # Enrich with timespan and serial filer stats
    def parse_dates(dates):
        return [d for d in dates if pd.notna(d)]

    groups['Timespan_Days'] = groups['IncorpDate'].apply(
        lambda dates: (max(dates) - min(dates)).days if len(parse_dates(dates)) > 1 else 0
    )

    groups['Filing_Timespan_Days'] = groups['filed_date'].apply(
        lambda dates: (max(dates) - min(dates)).days if len(parse_dates(dates)) > 1 else 0
    )

    groups['Serial_Filer_Count'] = groups['serial_filing'].apply(
        lambda x: sum(filing is True for filing in x)
    )

    groups['Serial_Filer_Pct'] = groups['Serial_Filer_Count'] / groups['LLC_Count']

    return groups

def main():
    parser = argparse.ArgumentParser(description="Group evictions data by specified fields.")
    parser.add_argument('fields', nargs='+', help='Fields to group by (e.g., RA-Name Street1 plaintiff_attorney)')
    args = parser.parse_args()

    # Load the input data
    input_file = '../OUTPUT/evictions_matched.parquet'
    try:
        df = pd.read_parquet(input_file)
    except FileNotFoundError:
        print(f"Input file {input_file} not found.")
        exit(1)

    # Convert date columns to datetime
    df['IncorpDate'] = pd.to_datetime(df['IncorpDate'], errors='coerce')
    df['filed_date'] = pd.to_datetime(df['filed_date'], errors='coerce')

    # Normalize the specified fields
    normalize_fields = args.fields
    missing_fields = [f for f in normalize_fields if f not in df.columns]
    if missing_fields:
        print(f"Error: The following fields are not in the DataFrame: {missing_fields}")
        exit(1)
    
    df = normalize_columns(df, normalize_fields)

    # Create normalized field names for composite key
    normalized_fields = [f'{field}_normalized' for field in normalize_fields]

    # Create composite key
    df, composite_key = create_composite_key(df, normalized_fields)

    # Analyze networks
    groups = analyze_llc_networks(df, composite_key)

    # Save results
    output_file = '../OUTPUT/evictions_grouped.parquet'
    groups.to_parquet(output_file)
    print(f"Grouped data saved to {output_file}")

if __name__ == '__main__':
    main()