import sys
import ollama
import argparse
import os
import time
import csv
import sqlite3

def query_ollama(model_name: str, problem_text: str, sys_prompt:str):
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
        description="Ask critical reasoning questions from a SQLite DB to a local LLM via Ollama and save results.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--model",
        required=True,
        help="The name of the Ollama model to use (e.g., 'llama3', 'mistral')."
    )

    parser.add_argument(
        "--range",
        help="0-indexed range of rows to process from the database, formatted as \"begin:end\" (e.g., \"0:99\"). Processes all rows if not specified."
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

    prompt_file_path = os.path.join(project_root, "data", "prompts", "critical_reasoning_prompt.txt")
    db_path = os.path.join(project_root, "data", "datasets", "critical_reasoning.db")
    output_csv_path = os.path.join(project_root, "data", "generated_data", "critical_reasoning_answers.csv")

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

    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
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
        with sqlite3.connect(db_path) as conn, open(output_csv_path, output_file_mode, newline='', encoding="utf-8") as outfile:
            cursor = conn.cursor()
            csv_writer = csv.DictWriter(outfile, fieldnames=output_header)

            if write_header_flag:
                csv_writer.writeheader()

            query = "SELECT QuestionID, QuestionText FROM Questions"
            if range_start is not None and range_end is not None:
                query += f" LIMIT {range_end - range_start + 1} OFFSET {range_start}"
            cursor.execute(query)
            rows = cursor.fetchall()

            processed_count = 0
            for i, row in enumerate(rows):
                current_row_index = i + (range_start or 0)
                try:
                    question_id = row[0]
                    problem_text = row[1]
                except IndexError as e:
                    print(f"Skipping row {current_row_index + 1} due to missing column: {e}. Expected 'QuestionID' and 'QuestionText'.")
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
                    print("\nNo questions processed. Check your --range values and database content.")
                else:
                    print("\nNo questions found or processed in the database.")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
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

