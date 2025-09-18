#!/usr/bin/env python3
"""
Script to clear outputs from all Jupyter notebooks in a directory.
Usage: python clear_notebook_outputs.py [directory_path]
If no directory is provided, defaults to current directory.
"""

import os
import sys
import json
from pathlib import Path

def clear_notebook_outputs(notebook_path):
    """Clear outputs from a single notebook file."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Clear outputs from all cells
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
        
        # Reset kernel metadata
        if 'metadata' in notebook:
            if 'kernelspec' in notebook['metadata']:
                notebook['metadata']['kernelspec'].pop('name', None)
            notebook['metadata'].pop('language_info', None)
        
        # Write back the cleaned notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Cleared outputs from: {notebook_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {notebook_path}: {e}")
        return False

def main():
    # Get directory from command line argument or use current directory
    if len(sys.argv) > 1:
        directory = Path(sys.argv[1])
    else:
        directory = Path('.')
    
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
    
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory.")
        sys.exit(1)
    
    # Find all .ipynb files
    notebook_files = list(directory.glob('**/*.ipynb'))
    
    if not notebook_files:
        print(f"No notebook files found in '{directory}'")
        return
    
    print(f"Found {len(notebook_files)} notebook(s) in '{directory}'")
    print("-" * 50)
    
    success_count = 0
    for notebook_file in notebook_files:
        if clear_notebook_outputs(notebook_file):
            success_count += 1
    
    print("-" * 50)
    print(f"Successfully cleared outputs from {success_count}/{len(notebook_files)} notebooks")

if __name__ == "__main__":
    main()
