import sys
import ollama
import argparse
import os
import time # For timing API calls
import csv # For CSV file operations

def query_ollama(model_name: str, problem_text: str, sys_prompt:str):
    """
    Queries the Ollama model with the given problem text and system prompt.

    Args:
        model_name: The name of the Ollama model to use (e.g., 'llama3', 'mistral').
        problem_text: The math problem to send to the model.
        sys_prompt: The system prompt to guide the model's behavior.

    Returns:
        A tuple containing the model's response (str) and the time taken (float).
        Returns (error_message, 0) if an error occurs.
    """
    try:
        start_time = time.time()
        response_data = ollama.generate(
            model=model_name,
            prompt=problem_text,
            system=sys_prompt
        )
        response_text = response_data['response']
        end_time = time.time()
        time_taken = end_time - start_time
        return response_text, time_taken
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return f"Error: {e}", 0

def main():
    """
    Main function to read math questions from a CSV, query Ollama, and save results.
    The script assumes it is located in a 'scripts' directory, and the 'data' directory
    is a sibling to 'scripts' at the project root.
    Project structure example:
    .
    ├── data
    │   ├── datasets
    │   │   └── math_questions_pool.csv
    │   ├── generated_data
    │   │   └── math_answers.csv
    │   └── prompts
    │       └── math_question_prompt.txt
    └── scripts
        └── math_answer_generator.py
    """
    parser = argparse.ArgumentParser(
        description="Ask math questions from a CSV to a local LLM via Ollama and save results.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--model",
        required=True,
        help="The name of the Ollama model to use (e.g., 'llama3', 'mistral')."
    )

    parser.add_argument(
        "--range",
        help="0-indexed range of rows to process from the input CSV, formatted as \"begin:end\" (e.g., \"0:99\"). Processes all rows if not specified."
    )

    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="If enabled, the output file is cleared and rewritten. Otherwise, results are appended."
    )

    args = parser.parse_args()
    model_name = args.model

    # Determine the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Determine the project root directory (assuming script is in 'scripts' subdirectory)
    project_root = os.path.dirname(script_dir)

    # Define paths relative to the project root
    prompt_file_path = os.path.join(project_root, "data", "prompts", "math_question_prompt.txt")
    input_csv_path = os.path.join(project_root, "data", "datasets", "math_questions_pool_1.csv")
    output_csv_path = os.path.join(project_root, "data", "generated_data", "math_answers.csv")
    
    # Define the header for the output CSV file
    output_header = ["uuid", "response", "model", "time_taken_seconds"] # Added "time_taken_seconds"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Load system prompt
    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            sys_prompt = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_file_path}")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading prompt file {prompt_file_path}: {e}")
        sys.exit(1)

    # Check if input CSV file exists
    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV file not found at {input_csv_path}")
        sys.exit(1)

    # Parse the range argument
    range_start, range_end = None, None
    if args.range:
        try:
            parts = args.range.split(':')
            if len(parts) != 2:
                raise ValueError("Range must have two parts separated by ':'")
            range_start = int(parts[0])
            range_end = int(parts[1])
            if range_start < 0 or range_end < 0 or range_start > range_end:
                raise ValueError("Invalid range values. Ensure begin <= end and both are non-negative.")
        except ValueError as e:
            print(f"Error: Invalid --range argument '{args.range}'. Format: 'begin:end'. {e}")
            sys.exit(1)
    
    # Determine file mode for output CSV
    output_file_mode = 'w' if args.reset_output else 'a'
    # Check if header needs to be written (if file is new, empty, or being reset)
    write_header_flag = False
    if output_file_mode == 'w':
        write_header_flag = True
    else: # Append mode
        if not os.path.exists(output_csv_path) or os.path.getsize(output_csv_path) == 0:
            write_header_flag = True
            
    try:
        with open(output_csv_path, output_file_mode, newline='', encoding="utf-8") as outfile:
            csv_writer = csv.DictWriter(outfile, fieldnames=output_header)

            if write_header_flag:
                csv_writer.writeheader()

            # Open and read the input CSV file
            with open(input_csv_path, "r", newline='', encoding="utf-8") as infile:
                csv_reader = csv.DictReader(infile)
                
                processed_count = 0
                for i, row in enumerate(csv_reader):
                    current_row_index = i # 0-indexed

                    # Apply range filter if specified
                    if range_start is not None and current_row_index < range_start:
                        continue
                    if range_end is not None and current_row_index > range_end:
                        print(f"\nReached end of specified range (row index {range_end}). Stopping.")
                        break
                    
                    try:
                        question_uuid = row['uuid']
                        problem_text = row['problem']
                    except KeyError as e:
                        print(f"Skipping row {current_row_index + 1} in {input_csv_path} due to missing column: {e}. Expected 'uuid' and 'problem'.")
                        continue

                    print(f"\nProcessing Question UUID: {question_uuid} (Row {current_row_index + 1})...")
                    
                    response_text, time_taken = query_ollama(model_name, problem_text, sys_prompt)

                    result_data = {
                        "uuid": question_uuid,
                        "response": response_text,
                        "model": model_name,
                        "time_taken_seconds": f"{time_taken:.1f}" # Added time_taken, formatted to 4 decimal places
                    }

                    if "Error:" in response_text and time_taken == 0: # Indicates an error from query_ollama
                        print(f"Failed to get response for UUID {question_uuid}: {response_text}")
                        # The error message is stored in response_text and will be written
                    else:
                        print(f"Response received in {time_taken:.2f} seconds.")
                        # print(f"Response snippet: {response_text[:100]}...") # Uncomment for quick check
                    
                    csv_writer.writerow(result_data)
                    processed_count += 1
                
                if processed_count == 0:
                    if range_start is not None or range_end is not None:
                        print("\nNo questions processed. Check your --range values and CSV content.")
                    else:
                        print("\nNo questions found or processed in the input CSV.")


    except FileNotFoundError:
        # This specific check is more for the output_csv_path if something goes wrong during its handling.
        # Input file existence is checked earlier.
        print(f"Error: A file operation failed. Please check paths and permissions for {output_csv_path}.")
        sys.exit(1)
    except IOError as e:
        print(f"An I/O error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    print(f"\nProcessing complete. Results saved to {output_csv_path}")

if __name__ == "__main__":
    main()
