import pandas as pd
from tqdm import tqdm
from ollama import chat, ChatResponse
from dotenv import load_dotenv
import json
from together import Together
import re
from enum import Enum
import os
import time
import sys
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# === CONFIG ===
MAX_RETRIES = 3
RETRY_DELAY = 5

def log(msg):
    """Print messages with timestamp and force flush for nohup."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

class QueryExecutor:
    class ExecType(Enum):
        NORMAL = 'NORMAL'
        REMOVE = 'REMOVE'
        NULLABLE = 'NULL'

    def __init__(self, remote_models: list, tests: str, local_models: list = [], 
                 exec_type: ExecType = ExecType.NORMAL, rem_columns: list = None):
        """
        Initialize QueryExecutor for CSV-based testing with Together AI.
        
        Args:
            remote_models: List of Together AI models to use
            tests: Path to test CSV file
            local_models: List of local models (optional)
            exec_type: Execution type for data modification
            rem_columns: Specific columns to remove/nullify
        """
        self.models = remote_models
        self.tests = pd.read_csv(tests)
        self.local_models = local_models
        self.exec_type = exec_type
        self.rem_columns = rem_columns or []
        
        # Initialize Together AI
        load_dotenv()
        API_KEY = os.getenv("TOGETHER_API_KEY")
        print(API_KEY)
        if not API_KEY:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        self.together = Together(api_key=API_KEY)
        
        log(f"Initialized QueryExecutor with {len(self.tests)} tests")
        log(f"Execution type: {exec_type.value}")
        log(f"Columns to modify: {self.rem_columns}")
        log(f"Remote models: {self.models}")

    def get_progress_file_path(self, output_path: str, model: str) -> str:
        """Generate progress file path based on output path and model."""
        base_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        model_safe = model.replace("/", "_").replace("-", "_")
        return os.path.join(base_dir, f"{base_name}_{model_safe}_progress.txt")

    def save_progress(self, progress_file: str, test_id: int):
        """Save current progress to file."""
        try:
            with open(progress_file, 'w') as f:
                f.write(str(test_id))
        except Exception as e:
            log(f"Warning: Could not save progress to {progress_file}: {e}")

    def load_progress(self, progress_file: str) -> int:
        """Load progress from file, return 0 if file doesn't exist."""
        try:
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress = int(f.read().strip())
                    log(f"Resuming from test_id: {progress}")
                    return progress
            return 0
        except Exception as e:
            log(f"Warning: Could not load progress from {progress_file}: {e}")
            return 0

    def initialize_output_file(self, output_path: str):
        """Initialize output CSV file with headers if it doesn't exist."""
        if not os.path.exists(output_path):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create header row
            required_columns = [
                'csv_path', 'test_language', 'test_category', 'ground_truth', 'nl_query', 
                'test_id', 'exec_type', 'processing_timestamp', 'success', 'error', 
                'predicted_ids', 'predicted_ids_str', 'raw_response', 'dataset_shape', 
                'retry_count', 'model'
            ]
            
            # Create empty DataFrame with required columns and save as CSV
            df_header = pd.DataFrame(columns=required_columns)
            df_header.to_csv(output_path, index=False)
            log(f"Initialized output file: {output_path}")

    def save_single_result(self, result: dict, output_path: str):
        """Save a single result to CSV file."""
        try:
            # Convert result to DataFrame
            result_df = pd.DataFrame([result])
            
            # Required columns in correct order
            required_columns = [
                'csv_path', 'test_language', 'test_category', 'ground_truth', 'nl_query', 
                'test_id', 'exec_type', 'processing_timestamp', 'success', 'error', 
                'predicted_ids', 'predicted_ids_str', 'raw_response', 'dataset_shape', 
                'retry_count', 'model'
            ]
            
            # Add missing columns if they don't exist
            for col in required_columns:
                if col not in result_df.columns:
                    result_df[col] = ''
            
            # Reorder columns
            other_columns = [col for col in result_df.columns if col not in required_columns]
            final_columns = required_columns + other_columns
            result_df = result_df[final_columns]
            
            # Append to CSV (without header since we already have it)
            result_df.to_csv(output_path, mode='a', header=False, index=False)
            
        except Exception as e:
            log(f"Error saving result to {output_path}: {e}")

    def is_test_already_processed(self, output_path: str, test_id: int, model: str) -> bool:
        """Check if a test has already been processed for a specific model."""
        try:
            if not os.path.exists(output_path):
                return False
            
            # Read existing results
            existing_df = pd.read_csv(output_path)
            
            # Check if this test_id and model combination exists
            exists = ((existing_df['test_id'] == test_id) & 
                     (existing_df['model'] == model)).any()
            
            return exists
        except Exception as e:
            log(f"Warning: Could not check if test is processed: {e}")
            return False

    def load_dataset(self, csv_path: str, exec_type: ExecType) -> pd.DataFrame:
        """
        Load dataset from CSV and apply modifications based on exec_type.

        Args:
            csv_path: path to CSV file
            exec_type: execution mode

        Returns:
            pandas.DataFrame: modified dataset
        """
        try:
            log(f"Loading dataset from {csv_path}")
            df = pd.read_csv(csv_path)

            if exec_type == QueryExecutor.ExecType.REMOVE:
                # Remove specified columns completely
                for col in self.rem_columns:
                    if col in df.columns:
                        df = df.drop(columns=[col], errors='ignore')
                        log(f"Removed column '{col}' from dataset")

            elif exec_type == QueryExecutor.ExecType.NULLABLE:
                # Set specified columns to NULL
                for col in self.rem_columns:
                    if col in df.columns:
                        df[col] = None
                        log(f"Set column '{col}' to NULL in dataset")

            elif exec_type == QueryExecutor.ExecType.NORMAL:
                log(f"No modifications applied to dataset")

            return df

        except Exception as e:
            log(f"Error loading dataset {csv_path}: {e}")
            return None

    def format_dataset_for_prompt(self, df: pd.DataFrame) -> str:
        """
        Format dataset for the prompt.

        Args:
            df: DataFrame of the dataset

        Returns:
            str: dataset formatted as JSON string
        """
        return df.to_json(orient='records')

    def create_prompt(self, dataset_str: str, nl_query: str) -> str:
        """
        Create the prompt for Together AI API.

        Args:
            dataset_str: formatted dataset string
            nl_query: natural language query

        Returns:
            str: complete prompt
        """
        prompt = f"""
You are a movie recommendation system. Analyze the provided movie dataset and respond to the user's request.

Dataset:
{dataset_str}

User Request:
{nl_query}

Instructions:
- Analyze the movies in the dataset
- For similarity requests, find movies with similar genres, themes, or characteristics
- For negative similarity requests ("opposto", "opposite", "contrario"), find movies that are opposite in style, genre, or theme
- Return the results as a list of movie IDs (tt codes) that best match the request
- Consider factors like genre, year, rating, plot, and other available attributes when available
- Provide exactly 10 recommendations when possible
- If some data is missing (NULL values), use the available information to make the best recommendations

Response Format:
Return only the movie IDs (tt codes) separated by |, for example:
tt0120338|tt0167260|tt0290334|tt0372784|tt0449088|tt0475783|tt0800080|tt0848228|tt1254207|tt1300854
"""
        return prompt

    def extract_movie_ids(self, response: str) -> list:
        """
        Extract movie IDs from API response.

        Args:
            response: API response

        Returns:
            list: list of movie IDs
        """
        # Look for tt pattern (tt followed by digits)
        tt_pattern = r'tt\d{7,8}'
        movie_ids = re.findall(tt_pattern, response)

        # If no tt pattern found, try to extract from formatted response
        if not movie_ids and '|' in response:
            movie_ids = [id.strip() for id in response.split('|') 
                        if id.strip() and (id.strip().startswith('tt') or id.strip().isdigit())]

        # Remove duplicates while maintaining order
        seen = set()
        unique_ids = []
        for id in movie_ids:
            if id not in seen:
                seen.add(id)
                unique_ids.append(id)

        return unique_ids

    def query_together_with_retry(self, prompt: str, model: str) -> Tuple[str, bool, str, int]:
        """
        Query Together AI API with automatic retry.

        Args:
            prompt: prompt to send
            model: model name

        Returns:
            Tuple[str, bool, str, int]: (response, success, error, retry_count)
        """
        retry_count = 0
        
        while retry_count <= MAX_RETRIES:
            try:
                log(f"Sending query to Together AI {model} (attempt {retry_count + 1})...")
                
                response = self.together.chat.completions.create(
                    temperature=0.1,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a highly skilled movie recommendation system. Always follow instructions carefully."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=model
                )
                
                log(f"Received response from Together AI")
                return response.choices[0].message.content.strip(), True, "", retry_count

            except Exception as e:
                error_msg = str(e)
                log(f"Error in Together AI query: {error_msg}")
                
                # Check if should retry
                retry_count += 1
                if "429" in error_msg and "rate limit" in error_msg.lower():
                    wait_time = 30
                    log(f"Rate limit hit, waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

                if retry_count <= MAX_RETRIES:
                    log(f"Retrying in {RETRY_DELAY} seconds... (attempt {retry_count + 1}/{MAX_RETRIES + 1})")
                    time.sleep(RETRY_DELAY)
                else:
                    return "", False, error_msg, retry_count
        
        return "", False, f"Maximum retries ({MAX_RETRIES}) reached", MAX_RETRIES

    def process_test_row(self, row: dict, exec_type: ExecType, model: str) -> dict:
        """
        Process a single test row.

        Args:
            row: test row dictionary
            exec_type: execution mode
            model: model name

        Returns:
            dict: test result
        """
        log(f"Processing test: {row.get('nl_query', 'Unknown query')[:50]}... (mode: {exec_type.value}, model: {model})")

        # Initialize result with all original columns
        result = row.copy()
        
        # Add new columns for results
        result['exec_type'] = exec_type.value
        result['processing_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result['model'] = model
        
        # Load dataset
        df = self.load_dataset(row['csv_path'], exec_type)
        if df is None:
            result.update({
                'success': False,
                'error': f"Unable to load dataset from {row['csv_path']}",
                'predicted_ids': [],
                'predicted_ids_str': '',
                'raw_response': '',
                'dataset_shape': '',
                'retry_count': 0
            })
            return result

        # Format dataset for prompt
        dataset_str = self.format_dataset_for_prompt(df)

        # Create prompt
        prompt = self.create_prompt(dataset_str, row['nl_query'])

        # Query Together AI with retry
        response, success, error_msg, retry_count = self.query_together_with_retry(prompt, model)
        
        if not success:
            result.update({
                'success': False,
                'error': error_msg,
                'predicted_ids': [],
                'predicted_ids_str': '',
                'raw_response': '',
                'dataset_shape': f"{df.shape[0]}x{df.shape[1]}",
                'retry_count': retry_count
            })
            return result

        # Extract movie IDs
        predicted_ids = self.extract_movie_ids(response)

        # Update result with all data
        result.update({
            'success': True,
            'error': '',
            'predicted_ids': predicted_ids,
            'predicted_ids_str': '|'.join(predicted_ids),
            'raw_response': '',
            'dataset_shape': f"{df.shape[0]}x{df.shape[1]}",
            'retry_count': retry_count
        })

        return result

    def execute_API_queries(self, output_path: str) -> pd.DataFrame:
        """Execute queries using Together AI with incremental saving."""
        log("Starting API queries with Together AI (incremental mode)")
        
        # Initialize output file
        self.initialize_output_file(output_path)
        
        # Add test_id if not present
        if 'test_id' not in self.tests.columns:
            self.tests['test_id'] = range(len(self.tests))
        
        for model in self.models:
            log(f"Processing with model: {model}")
            
            # Get progress file path
            progress_file = self.get_progress_file_path(output_path, model)
            
            # Load progress
            last_completed_test_id = self.load_progress(progress_file)
            
            # Filter tests to process (only those not yet completed)
            tests_to_process = self.tests[self.tests['test_id'] > last_completed_test_id]
            
            if len(tests_to_process) == 0:
                log(f"All tests already completed for model {model}")
                continue
            
            log(f"Processing {len(tests_to_process)} tests for model {model}")
            
            for idx, row in tqdm(tests_to_process.iterrows(), total=len(tests_to_process), 
                               colour="green", desc=f"API_queries on model: {model}"):
                
                test_id = row['test_id']
                
                # Skip if already processed
                if self.is_test_already_processed(output_path, test_id, model):
                    log(f"Test {test_id} already processed for model {model}, skipping")
                    continue
                
                try:
                    # Process the test
                    result = self.process_test_row(row.to_dict(), self.exec_type, model)
                    result['test_id'] = test_id
                    
                    # Save result immediately
                    self.save_single_result(result, output_path)
                    
                    # Update progress
                    self.save_progress(progress_file, test_id)
                    
                    if result['success']:
                        log(f"Successfully processed test {test_id}: {len(result['predicted_ids'])} IDs extracted")
                    else:
                        log(f"Failed test {test_id}: {result['error']}")
                    
                    # Small delay to avoid rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    log(f"Error processing test {test_id}: {e}")
                    
                    # Create error result and save it
                    error_result = row.to_dict()
                    error_result.update({
                        'test_id': test_id,
                        'success': False,
                        'error': str(e),
                        'exec_type': self.exec_type.value,
                        'predicted_ids': [],
                        'predicted_ids_str': '',
                        'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'retry_count': MAX_RETRIES,
                        'raw_response': '',
                        'dataset_shape': '',
                        'model': model
                    })
                    
                    self.save_single_result(error_result, output_path)
                    self.save_progress(progress_file, test_id)
            
            # Clean up progress file when model is complete
            try:
                if os.path.exists(progress_file):
                    os.remove(progress_file)
                    log(f"Completed all tests for model {model}, removed progress file")
            except Exception as e:
                log(f"Warning: Could not remove progress file: {e}")
        
        # Return final results
        try:
            return pd.read_csv(output_path)
        except Exception as e:
            log(f"Error reading final results: {e}")
            return pd.DataFrame()

    def execute_local_query(self, output_path: str) -> pd.DataFrame:
        """Execute queries using local models (Ollama) with incremental saving."""
        log("Starting local queries (incremental mode)")
        
        # Initialize output file
        self.initialize_output_file(output_path)
        
        # Add test_id if not present
        if 'test_id' not in self.tests.columns:
            self.tests['test_id'] = range(len(self.tests))
        
        for model in self.local_models:
            log(f"Processing with local model: {model}")
            model_name = f"{model}-local"
            
            # Get progress file path
            progress_file = self.get_progress_file_path(output_path, model_name)
            
            # Load progress
            last_completed_test_id = self.load_progress(progress_file)
            
            # Filter tests to process
            tests_to_process = self.tests[self.tests['test_id'] > last_completed_test_id]
            
            if len(tests_to_process) == 0:
                log(f"All tests already completed for model {model_name}")
                continue
            
            for idx, row in tqdm(tests_to_process.iterrows(), total=len(tests_to_process),
                               colour="blue", desc=f"Local_queries on model: {model}"):
                
                test_id = row['test_id']
                
                # Skip if already processed
                if self.is_test_already_processed(output_path, test_id, model_name):
                    log(f"Test {test_id} already processed for model {model_name}, skipping")
                    continue
                
                # Load dataset
                df = self.load_dataset(row['csv_path'], self.exec_type)
                if df is None:
                    continue
                    
                dataset_str = self.format_dataset_for_prompt(df)
                prompt = self.create_prompt(dataset_str, row['nl_query'])
                
                try:
                    response: ChatResponse = chat(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a highly skilled movie recommendation system. Always follow instructions carefully."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )
                    
                    predicted_ids = self.extract_movie_ids(response['message']['content'])
                    
                    result = row.to_dict()
                    result.update({
                        'test_id': test_id,
                        'exec_type': self.exec_type.value,
                        'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'success': True,
                        'error': '',
                        'predicted_ids': predicted_ids,
                        'predicted_ids_str': '|'.join(predicted_ids),
                        'raw_response': '',
                        'dataset_shape': f"{df.shape[0]}x{df.shape[1]}",
                        'retry_count': 0,
                        'model': model_name
                    })
                    
                    # Save result immediately
                    self.save_single_result(result, output_path)
                    self.save_progress(progress_file, test_id)
                    
                except Exception as e:
                    log(f"Error in local query for {model}: {e}")
                    result = row.to_dict()
                    result.update({
                        'test_id': test_id,
                        'exec_type': self.exec_type.value,
                        'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'success': False,
                        'error': str(e),
                        'predicted_ids': [],
                        'predicted_ids_str': '',
                        'raw_response': '',
                        'dataset_shape': f"{df.shape[0]}x{df.shape[1]}" if df is not None else '',
                        'retry_count': 0,
                        'model': model_name
                    })
                    
                    self.save_single_result(result, output_path)
                    self.save_progress(progress_file, test_id)
            
            # Clean up progress file when model is complete
            try:
                if os.path.exists(progress_file):
                    os.remove(progress_file)
                    log(f"Completed all tests for model {model_name}, removed progress file")
            except Exception as e:
                log(f"Warning: Could not remove progress file: {e}")
        
        # Return final results
        try:
            return pd.read_csv(output_path)
        except Exception as e:
            log(f"Error reading final results: {e}")
            return pd.DataFrame()

    def run(self, output_path: str) -> pd.DataFrame:
        """Run all queries and return results with incremental saving."""
        log("Starting QueryExecutor run (incremental mode)")
        
        # Execute API queries (Together AI)
        result = self.execute_API_queries(output_path)

        # Execute local queries if specified
        if self.local_models:
            local_result = self.execute_local_query(output_path)
            # Combine results if both exist
            if len(result) > 0 and len(local_result) > 0:
                result = pd.concat([result, local_result], ignore_index=True)
            elif len(local_result) > 0:
                result = local_result

        log(f"QueryExecutor run completed. Total results: {len(result)}")
        return result

    def run_tests_from_csv(self, output_dir: str = "results", exec_types: list = None):
        """
        Run tests from CSV with different execution types using incremental saving.

        Args:
            output_dir: output directory for results
            exec_types: list of execution modes to test
        """
        if exec_types is None:
            exec_types = [self.exec_type]
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Add test_id if not present
        if 'test_id' not in self.tests.columns:
            self.tests['test_id'] = range(len(self.tests))
        
        for exec_type in exec_types:
            log(f"Starting tests in mode {exec_type.value}")
            
            # Set the execution type
            original_exec_type = self.exec_type
            self.exec_type = exec_type
            
            # Define output path for this mode
            model_name = "_".join(self.models).replace("/", "_").replace("-", "_")
            output_path = os.path.join(output_dir, f"results_{exec_type.value.lower()}_{model_name}.csv")
            
            try:
                # Run tests with incremental saving
                results = self.run(output_path)
                
                # Print statistics
                if len(results) > 0:
                    total_tests = len(results)
                    successful_tests = len(results[results['success'] == True])
                    log(f"Completed tests for mode {exec_type.value}")
                    log(f"Total tests: {total_tests}")
                    log(f"Successful tests: {successful_tests}/{total_tests}")
                    
                    if successful_tests > 0:
                        avg_ids = results[results['success'] == True]['predicted_ids'].apply(len).mean()
                        log(f"Average predicted IDs per successful test: {avg_ids:.1f}")
                else:
                    log(f"No results found for mode {exec_type.value}")
                
            finally:
                # Restore original execution type
                self.exec_type = original_exec_type

# Example usage
if __name__ == "__main__":
    # Example configuration
    remote_models = ["meta-llama/Llama-2-7b-chat-hf"]
    test_csv = "path/to/tests.csv"
    local_models = []  # Optional
    exec_types = [QueryExecutor.ExecType.NORMAL, QueryExecutor.ExecType.REMOVE, QueryExecutor.ExecType.NULLABLE]
    rem_columns = ['title', 'year']  # Columns to modify
    
    # Initialize and run
    executor = QueryExecutor(
        remote_models=remote_models,
        tests=test_csv,
        local_models=local_models,
        exec_type=QueryExecutor.ExecType.NORMAL,  # Default, will be overridden
        rem_columns=rem_columns
    )
     
    # Run tests for all execution types
    executor.run_tests_from_csv("results", exec_types)
