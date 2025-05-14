import json
import sys
import os

def convert_ipynb_to_py(notebook_path, output_path=None):
    # If no output path is provided, replace .ipynb with .py
    if output_path is None:
        output_path = os.path.splitext(notebook_path)[0] + '.py'

    # Read the notebook file
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except FileNotFoundError:
        print(f"Error: Notebook file '{notebook_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{notebook_path}' is not a valid Jupyter Notebook.")
        sys.exit(1)

    # Open the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        # Iterate through cells
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                # Write code cell content
                for line in cell.get('source', []):
                    f.write(line)
                    if not line.endswith('\n'):
                        f.write('\n')
                f.write('\n')  # Add extra newline between cells

    print(f"Successfully converted '{notebook_path}' to '{output_path}'")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python nbconvert.py <notebook.ipynb> [output.py]")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert_ipynb_to_py(notebook_path, output_path)