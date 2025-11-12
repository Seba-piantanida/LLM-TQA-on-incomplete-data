#!/usr/bin/env python3
"""
CSV Random Row Sampler

This script takes a CSV file and randomly samples n rows from it,
saving the result to a new file with the naming convention [input_file]_cut_n.csv
"""

import pandas as pd
import argparse
import os
import sys
from pathlib import Path

def log(msg):
    """Print messages with timestamp."""
    print(f"{msg}")
    sys.stdout.flush()

def sample_csv_rows(input_file: str, n: int, seed: int = None) -> str:
    """
    Sample n random rows from a CSV file and save to a new file.
    
    Args:
        input_file: Path to input CSV file
        n: Number of rows to sample
        seed: Random seed for reproducibility (optional)
        
    Returns:
        str: Path to the output file
    """
    try:
        # Load the CSV file
        log(f"Loading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        
        original_rows = len(df)
        log(f"Original file contains {original_rows} rows")
        
        # Clean the dataset by removing rows with missing values in key columns
        df = df.dropna(subset=['title', 'year', 'genre', 'directors'])
        log(f"✅ Dataset pulito: {len(df)} film validi")
        
        # Check if n is valid
        if n <= 0:
            raise ValueError("Number of rows to sample must be positive")
        
        if n > original_rows:
            log(f"Warning: Requested {n} rows but file only has {original_rows} rows")
            log("Using all available rows")
            n = original_rows
            sampled_df = df.copy()
        else:
            # Sample n random rows
            if seed is not None:
                log(f"Using random seed: {seed}")
            
            sampled_df = df.sample(n=n, random_state=seed)
            log(f"Sampled {n} random rows")
        
        # Generate output filename
        input_path = Path(input_file)
        stem = input_path.stem  # filename without extension
        suffix = input_path.suffix  # .csv
        parent = input_path.parent
        
        output_filename = f"{stem}_cut_{n}{suffix}"
        output_path = parent / output_filename
        
        # Save the sampled data
        sampled_df.to_csv(output_path, index=False)
        log(f"Sampled data saved to: {output_path}")
        
        return str(output_path)
        
    except FileNotFoundError:
        log(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        log(f"Error: The input file is empty: {input_file}")
        sys.exit(1)
    except pd.errors.ParserError as e:
        log(f"Error: Could not parse CSV file: {e}")
        sys.exit(1)
    except ValueError as e:
        log(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        log(f"Unexpected error: {e}")
        sys.exit(1)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Randomly sample n rows from a CSV file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python csv_sampler.py input.csv 100
    python csv_sampler.py data/test.csv 500 --seed 42
    python csv_sampler.py /path/to/file.csv 50 --seed 123
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        'n',
        type=int,
        help='Number of rows to sample'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducible sampling (optional)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        log(f"Error: Input file does not exist: {args.input_file}")
        sys.exit(1)
    
    # Validate n is positive
    if args.n <= 0:
        log("Error: Number of rows must be positive")
        sys.exit(1)
    
    if args.verbose:
        log(f"Input file: {args.input_file}")
        log(f"Number of rows to sample: {args.n}")
        if args.seed is not None:
            log(f"Random seed: {args.seed}")
    
    try:
        output_file = sample_csv_rows(args.input_file, args.n, args.seed)
        
        if args.verbose:
            # Show some statistics
            input_df = pd.read_csv(args.input_file)
            output_df = pd.read_csv(output_file)
            
            log("\nSampling completed successfully!")
            log(f"Original rows: {len(input_df)}")
            log(f"Sampled rows: {len(output_df)}")
            log(f"Sampling ratio: {len(output_df)/len(input_df)*100:.1f}%")
            log(f"Output file: {output_file}")
        else:
            log("Sampling completed successfully!")
            
    except KeyboardInterrupt:
        log("\nInterrupted by user. Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()