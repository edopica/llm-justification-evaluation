import sys
import ollama
import argparse
import os
import time  # For timing API calls
import csv  # For CSV file operations

def query_ollama(model_name: str, problem_text: str, sys_prompt: str):
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
    parser = argparse.ArgumentParser(
        description="Ask essay evaluation questions from a CSV to a local LLM via Ollama and save results.",
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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    prompt_file_path = os.path.join(project_root, "data", "prompts", "essay_evaluation_prompt.txt")
    input_csv_path = os.path.join(project_root, "data", "datasets", "ielts_essays_questions.csv")
    output_csv_path = os.path.join(project_root, "data", "generated_data", "essay_evaluation_answers.csv")

    output_header = ["QuestionID", "response", "model", "time_taken_seconds"]

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            sys_prompt = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_file_path}")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading prompt file {prompt_file_path}: {e}")
        sys.exit(1)

    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV file not found at {input_csv_path}")
        sys.exit(1)

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

    output_file_mode = 'w' if args.reset_output else 'a'
    write_header_flag = False
    if output_file_mode == 'w':
        write_header_flag = True
    else:
        if not os.path.exists(output_csv_path) or os.path.getsize(output_csv_path) == 0:
            write_header_flag = True

    try:
        with open(output_csv_path, output_file_mode, newline='', encoding="utf-8") as outfile:
            csv_writer = csv.DictWriter(outfile, fieldnames=output_header)

            if write_header_flag:
                csv_writer.writeheader()

            with open(input_csv_path, "r", newline='', encoding="utf-8") as infile:
                csv_reader = csv.DictReader(infile)

                processed_count = 0
                for i, row in enumerate(csv_reader):
                    current_row_index = i

                    if range_start is not None and current_row_index < range_start:
                        continue
                    if range_end is not None and current_row_index > range_end:
                        print(f"\nReached end of specified range (row index {range_end}). Stopping.")
                        break

                    try:
                        question_id = row['QuestionID']
                        prompt_text = row['prompt']
                        essay_text = row['essay']
                        problem_text = f"Prompt: {prompt_text}\n\nEssay: {essay_text}"
                    except KeyError as e:
                        print(f"Skipping row {current_row_index + 1} in {input_csv_path} due to missing column: {e}. Expected 'QuestionID', 'prompt', 'essay'.")
                        continue

                    print(f"\nProcessing QuestionID: {question_id} (Row {current_row_index + 1})...")

                    response_text, time_taken = query_ollama(model_name, problem_text, sys_prompt)

                    result_data = {
                        "QuestionID": question_id,
                        "response": response_text,
                        "model": model_name,
                        "time_taken_seconds": f"{time_taken:.1f}"
                    }

                    if "Error:" in response_text and time_taken == 0:
                        print(f"Failed to get response for QuestionID {question_id}: {response_text}")
                    else:
                        print(f"Response received in {time_taken:.2f} seconds.")

                    csv_writer.writerow(result_data)
                    processed_count += 1

                if processed_count == 0:
                    if range_start is not None or range_end is not None:
                        print("\nNo questions processed. Check your --range values and CSV content.")
                    else:
                        print("\nNo questions found or processed in the input CSV.")

    except FileNotFoundError:
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

