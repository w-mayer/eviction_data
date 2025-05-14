import argparse
import subprocess
import os

def pull_eviction_data(data_dir):
    """Run pull_from_box.py to download eviction.csv."""
    try:
        subprocess.run(['python', 'pull_from_box.py', data_dir], check=True)
        print("Successfully pulled eviction.csv")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling eviction data: {e}")
        exit(1)

def fix_llc_csv(llc_csv):
    """Run fix_csv.py on LLC.csv."""
    if not os.path.exists(llc_csv):
        print(f"LLC.csv not found at {llc_csv}. Ensure it is downloaded.")
        exit(1)
    try:
        subprocess.run(['python', 'fix_csv.py', llc_csv], check=True)
        print(f"Successfully fixed {llc_csv}")
    except subprocess.CalledProcessError as e:
        print(f"Error fixing LLC.csv: {e}")
        exit(1)
    return llc_csv

def run_eviction_matching(eviction_csv, llc_csv):
    """Run eviction_matching.py with eviction.csv and LLC.csv."""
    if not os.path.exists(eviction_csv):
        print(f"eviction.csv not found at {eviction_csv}. Run with --pull-data first.")
        exit(1)
    if not os.path.exists(llc_csv):
        print(f"LLC.csv not found at {llc_csv}. Ensure it is downloaded.")
        exit(1)
    try:
        subprocess.run(['python', 'eviction_matching.py', eviction_csv, llc_csv], check=True)
        print("Successfully ran eviction matching")
    except subprocess.CalledProcessError as e:
        print(f"Error running eviction matching: {e}")
        exit(1)
        
def main():
    parser = argparse.ArgumentParser(description="Evictions Analysis Pipeline")
    parser.add_argument('--pull-data', action='store_true', help='Pull new eviction data (eviction.csv)')
    parser.add_argument('--run-matching', action='store_true', help='Run eviction matching with eviction.csv and LLC.csv')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing LLC.csv')
    args = parser.parse_args()

    # Construct file paths
    eviction_csv = os.path.join(args.data_dir, 'cases_residential_only.txt')
    llc_csv = os.path.join(args.data_dir, 'LLC.csv')

    # Step 1: Pull Eviction Data
    if args.pull_data:
        pull_eviction_data(args.data_dir)

    # Step 3: Run Eviction Matching
    if args.run_matching:
        fix_llc_csv(llc_csv)
        run_eviction_matching(eviction_csv, llc_csv)

if __name__ == '__main__':
    main()