import csv

original_file = '../DATA/LLC.csv'
cleaned_file = '../DATA/LLC_clean.csv'

with open(original_file, 'r', newline='') as infile, open(cleaned_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Write the header, skipping the leading empty field
    header = next(reader)
    writer.writerow(header[1:])  # e.g., ['EntityID', 'Name', 'Status', ...]
    
    # Process each data row
    for row in reader:
        cleaned_row = [field.strip() for field in row]  # Clean whitespace
        if len(cleaned_row) == 32 and cleaned_row[0] == '':
            cleaned_row = cleaned_row[1:]  # Remove leading empty field
        elif len(cleaned_row) < 31:
            cleaned_row += [''] * (31 - len(cleaned_row))  # Pad with empty strings
        elif len(cleaned_row) > 31:
            cleaned_row = cleaned_row[:31]  # Truncate extra fields
        writer.writerow(cleaned_row)

