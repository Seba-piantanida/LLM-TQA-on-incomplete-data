#!/usr/bin/env python3
"""
Query Runner Script with JSON Configuration and Incremental Saving

This script reads configuration from a JSON file and runs tests using the QueryExecutor class
with incremental saving and resume functionality. It supports multiple models, test files,
and execution types as specified in the JSON config.
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Import the QueryExecutor class (assuming it's in the same directory or installed as a module)
try:
    from query_executor_similarity import QueryExecutor
except ImportError:
    print("Error: Could not import QueryExecutor class. Make sure query_executor.py is in the same directory.")
    sys.exit(1)

def log(msg):
    """Print messages with timestamp and force flush."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def load_config(config_path: str) -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        log(f"Loaded configuration from {config_path}")
        print(config)
        return config
    except FileNotFoundError:
        log(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        log(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        log(f"Error loading configuration: {e}")
        sys.exit(1)

def validate_config(config: dict) -> bool:
    """
    Validate the configuration structure.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_keys = ['models', 'tests']
    
    # Check required top-level keys
    for key in required_keys:
        if key not in config:
            log(f"Error: Missing required key '{key}' in configuration")
            return False
    
    # Validate models
    if not isinstance(config['models'], list) or not config['models']:
        log("Error: 'models' must be a non-empty list")
        return False
    
    # Validate tests
    if not isinstance(config['tests'], list) or not config['tests']:
        log("Error: 'tests' must be a non-empty list")
        return False
    
    # Validate each test configuration
    for i, test in enumerate(config['tests']):
        required_test_keys = ['test_file', 'out_path', 'test_types']
        
        for key in required_test_keys:
            if key not in test:
                log(f"Error: Missing required key '{key}' in test {i}")
                return False
        
        # Check if test file exists
        if not os.path.exists(test['test_file']):
            log(f"Warning: Test file does not exist: {test['test_file']}")
        
        # Validate test types
        valid_test_types = ['normal', 'remove', 'nullable']
        for test_type in test['test_types']:
            if test_type.lower() not in valid_test_types:
                log(f"Error: Invalid test type '{test_type}' in test {i}. Valid types: {valid_test_types}")
                return False
    
    return True

def get_exec_type(test_type_str: str) -> QueryExecutor.ExecType:
    """
    Convert string test type to ExecType enum.
    
    Args:
        test_type_str: String representation of test type
        
    Returns:
        QueryExecutor.ExecType: Corresponding enum value
    """
    type_mapping = {
        'normal': QueryExecutor.ExecType.NORMAL,
        'remove': QueryExecutor.ExecType.REMOVE,
        'nullable': QueryExecutor.ExecType.NULLABLE
    }
    return type_mapping[test_type_str.lower()]

def ensure_output_dir(output_path: str):
    """
    Ensure output directory exists.
    
    Args:
        output_path: Path to output directory
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    log(f"Ensured output directory exists: {output_path}")

def get_existing_progress_info(output_file_path: str, models: list) -> dict:
    """
    Get information about existing progress for all models.
    
    Args:
        output_file_path: Path to the output CSV file
        models: List of model names
        
    Returns:
        dict: Progress information for each model
    """
    progress_info = {}
    
    if not os.path.exists(output_file_path):
        for model in models:
            progress_info[model] = {'completed': 0, 'total': 'unknown'}
        return progress_info
    
    try:
        import pandas as pd
        existing_df = pd.read_csv(output_file_path)
        
        for model in models:
            completed_tests = len(existing_df[existing_df['model'] == model])
            progress_info[model] = {'completed': completed_tests, 'total': 'unknown'}
    except Exception as e:
        log(f"Warning: Could not read existing progress from {output_file_path}: {e}")
        for model in models:
            progress_info[model] = {'completed': 0, 'total': 'unknown'}
    
    return progress_info

def estimate_remaining_time(start_time: datetime, completed: int, total: int) -> str:
    """
    Estimate remaining time based on current progress.
    
    Args:
        start_time: When the process started
        completed: Number of completed tests
        total: Total number of tests
        
    Returns:
        str: Formatted time estimate
    """
    if completed == 0 or total == 0:
        return "unknown"
    
    elapsed_time = datetime.now() - start_time
    time_per_test = elapsed_time.total_seconds() / completed
    remaining_tests = total - completed
    remaining_seconds = time_per_test * remaining_tests
    
    if remaining_seconds < 60:
        return f"{remaining_seconds:.0f} seconds"
    elif remaining_seconds < 3600:
        return f"{remaining_seconds/60:.0f} minutes"
    else:
        return f"{remaining_seconds/3600:.1f} hours"

def run_test_configuration(config: dict, dry_run: bool = False, resume: bool = True):
    """
    Run all tests specified in the configuration with incremental saving.
    
    Args:
        config: Configuration dictionary
        dry_run: If True, only validate configuration without running tests
        resume: If True, resume from existing progress
    """
    models = config['models']
    tests = config['tests']
    
    log(f"Configuration loaded:")
    log(f"  Models: {models}")
    log(f"  Number of test configurations: {len(tests)}")
    log(f"  Resume mode: {'enabled' if resume else 'disabled'}")
    
    total_runs = sum(len(test['test_types']) for test in tests)
    log(f"  Total test runs planned: {total_runs}")
    
    if dry_run:
        log("Dry run mode - configuration validated successfully")
        return
    
    # Counter for progress tracking
    current_run = 0
    overall_start_time = datetime.now()
    
    for test_config in tests:
        test_file = test_config['test_file']
        out_path = test_config['out_path']
        test_types = test_config['test_types']
        # Get rem_col from the specific test config, default to empty list if not present
        rem_columns = test_config.get('rem_col', [])
        
        log(f"\n" + "="*80)
        log(f"Processing test file: {test_file}")
        log(f"Output path: {out_path}")
        log(f"Test types: {test_types}")
        if rem_columns:
            log(f"Columns to modify: {rem_columns}")
        log("="*80)
        
        # Ensure output directory exists
        ensure_output_dir(out_path)
        
        # Convert test types to ExecType enums
        exec_types = [get_exec_type(t) for t in test_types]
        
        try:
            # Initialize QueryExecutor
            executor = QueryExecutor(
                remote_models=models,
                tests=test_file,
                local_models=[],  # No local models for now
                exec_type=QueryExecutor.ExecType.NORMAL,  # Default, will be overridden
                rem_columns=rem_columns
            )
            
            # Get total number of tests for progress calculation
            try:
                import pandas as pd
                test_df = pd.read_csv(test_file)
                total_tests_per_type = len(test_df)
            except Exception:
                total_tests_per_type = 0
            
            # Run tests for each execution type
            for exec_type in exec_types:
                current_run += 1
                test_start_time = datetime.now()
                
                log(f"\n[{current_run}/{total_runs}] Running test type: {exec_type.value}")
                log(f"Expected tests per model: {total_tests_per_type}")
                
                # Generate output filename
                exec_type_name = exec_type.value.lower()
                model_names = "_".join([m.split("/")[-1].replace("-", "_") for m in models])  # Use only model names, not full paths
                output_filename = f"results_{exec_type_name}_{model_names}.csv"
                output_file_path = os.path.join(out_path, output_filename)
                
                # Check existing progress if resume is enabled
                if resume:
                    progress_info = get_existing_progress_info(output_file_path, models)
                    log("Existing progress:")
                    for model, info in progress_info.items():
                        model_short = model.split("/")[-1]
                        log(f"  {model_short}: {info['completed']} tests completed")
                    
                    total_existing = sum(info['completed'] for info in progress_info.values())
                    total_expected = total_tests_per_type * len(models)
                    if total_existing > 0:
                        completion_pct = (total_existing / total_expected) * 100 if total_expected > 0 else 0
                        log(f"  Overall progress: {total_existing}/{total_expected} ({completion_pct:.1f}%)")
                
                # Set the execution type
                executor.exec_type = exec_type
                
                # Run the tests with incremental saving
                log(f"Starting execution with incremental saving...")
                results = executor.run(output_file_path)
                
                # Calculate and display final statistics
                if len(results) > 0:
                    # Group by model for detailed statistics
                    total_tests = len(results)
                    successful_tests = len(results[results['success'] == True])
                    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
                    
                    log(f"\n  Final Results for {exec_type.value}:")
                    log(f"  Output file: {output_file_path}")
                    log(f"  Total tests: {total_tests}")
                    log(f"  Successful tests: {successful_tests}")
                    log(f"  Success rate: {success_rate:.1f}%")
                    
                    # Per-model statistics
                    for model in models:
                        model_results = results[results['model'] == model]
                        if len(model_results) > 0:
                            model_successful = len(model_results[model_results['success'] == True])
                            model_success_rate = (model_successful / len(model_results)) * 100
                            model_short = model.split("/")[-1]
                            log(f"    {model_short}: {model_successful}/{len(model_results)} ({model_success_rate:.1f}%)")
                    
                    if successful_tests > 0:
                        # Calculate average predicted IDs per successful test
                        avg_ids = results[results['success'] == True]['predicted_ids'].apply(len).mean()
                        log(f"  Average predicted IDs per successful test: {avg_ids:.1f}")
                    
                    # Time statistics
                    test_duration = datetime.now() - test_start_time
                    log(f"  Test duration: {test_duration}")
                    if total_tests > 0:
                        avg_time_per_test = test_duration.total_seconds() / total_tests
                        log(f"  Average time per test: {avg_time_per_test:.1f} seconds")
                else:
                    log(f"  No results generated for {exec_type.value}")
                
                # Estimate overall remaining time
                overall_elapsed = datetime.now() - overall_start_time
                remaining_runs = total_runs - current_run
                if current_run > 0 and remaining_runs > 0:
                    avg_time_per_run = overall_elapsed.total_seconds() / current_run
                    estimated_remaining = avg_time_per_run * remaining_runs
                    if estimated_remaining < 3600:
                        time_str = f"{estimated_remaining/60:.0f} minutes"
                    else:
                        time_str = f"{estimated_remaining/3600:.1f} hours"
                    log(f"  Estimated remaining time: {time_str}")
        
        except Exception as e:
            log(f"Error processing test configuration: {e}")
            import traceback
            log(f"Traceback: {traceback.format_exc()}")
            log(f"Continuing to next test configuration...")
            continue
    
    total_duration = datetime.now() - overall_start_time
    log(f"\n" + "="*80)
    log("All test configurations completed!")
    log(f"Total execution time: {total_duration}")
    log("="*80)

def clean_progress_files(config: dict, output_base_dir: str = None):
    """
    Clean up progress files from previous runs.
    
    Args:
        config: Configuration dictionary
        output_base_dir: Base output directory (optional)
    """
    log("Cleaning up progress files...")
    
    models = config['models']
    tests = config['tests']
    cleaned_count = 0
    
    for test_config in tests:
        out_path = test_config['out_path']
        test_types = test_config['test_types']
        
        for test_type in test_types:
            exec_type_name = test_type.lower()
            model_names = "_".join([m.split("/")[-1].replace("-", "_") for m in models])
            output_filename = f"results_{exec_type_name}_{model_names}.csv"
            output_file_path = os.path.join(out_path, output_filename)
            
            # Look for progress files
            base_dir = os.path.dirname(output_file_path)
            base_name = os.path.splitext(os.path.basename(output_file_path))[0]
            
            for model in models:
                model_safe = model.replace("/", "_").replace("-", "_")
                progress_file = os.path.join(base_dir, f"{base_name}_{model_safe}_progress.txt")
                
                if os.path.exists(progress_file):
                    try:
                        os.remove(progress_file)
                        cleaned_count += 1
                        log(f"  Removed: {progress_file}")
                    except Exception as e:
                        log(f"  Warning: Could not remove {progress_file}: {e}")
    
    log(f"Cleaned up {cleaned_count} progress files")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Run query tests using JSON configuration with incremental saving',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python query_runner.py --config config.json
    python query_runner.py --config config.json --dry-run
    python query_runner.py --config config.json --no-resume
    python query_runner.py --config config.json --clean-progress
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without running tests'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable resume functionality (start from beginning)'
    )
    
    parser.add_argument(
        '--clean-progress',
        action='store_true',
        help='Clean up progress files and exit'
    )
    
    args = parser.parse_args()
    
    # Load and validate configuration
    config = load_config(args.config)
    
    # Handle progress file cleanup
    if args.clean_progress:
        clean_progress_files(config)
        log("Progress files cleaned. Exiting.")
        return
    
    # Run tests
    try:
        resume_mode = not args.no_resume
        run_test_configuration(config, args.dry_run, resume_mode)
    except KeyboardInterrupt:
        log("\nInterrupted by user. Progress has been saved and can be resumed later.")
        log("Run the script again to resume from where it left off.")
        sys.exit(0)
    except Exception as e:
        log(f"Unexpected error: {e}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()