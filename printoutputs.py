import json
import sys

def print_notebook_outputs(notebook_path):
    """
    Reads a Jupyter notebook and prints the outputs of each cell.
    If a cell has no output, prints a message indicating that.
    
    Args:
        notebook_path (str): Path to the .ipynb file
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find notebook at {notebook_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: {notebook_path} is not a valid Jupyter notebook")
        return

    # Get cells from the notebook
    cells = notebook.get('cells', [])
    
    if not cells:
        print("No cells found in notebook")
        return

    for idx, cell in enumerate(cells, 1):
        print(f"\n=== Cell {idx} ===")
        
        # Skip non-code cells
        if cell['cell_type'] != 'code':
            print(f"[Not a code cell - {cell['cell_type']}]")
            continue
            
        outputs = cell.get('outputs', [])
        
        if not outputs:
            print(f"No output in cell {idx}")
            continue
            
        for output_idx, output in enumerate(outputs, 1):
            # Handle different types of output
            if 'text' in output:
                print(f"Text output {output_idx}:")
                print(output['text'])
            elif 'data' in output:
                data = output['data']
                if 'text/plain' in data:
                    print(f"Plain text output {output_idx}:")
                    print(data['text/plain'])
                # You might want to handle other types like 'image/png', etc.
            elif 'ename' in output:
                # Handle error output
                print(f"Error output {output_idx}:")
                print(f"{output.get('ename')}: {output.get('evalue')}")
            else:
                print(f"Unknown output type in output {output_idx}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_notebook.ipynb>")
        sys.exit(1)
    
    print_notebook_outputs(sys.argv[1])
