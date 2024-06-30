import json
import os

def aggregate_logs(log_file_path):
    # Create the log file if it doesn't exist
    if not os.path.exists(log_file_path):
        open(log_file_path, 'w').close()
        print(f"Log file {log_file_path} created.")

    log_entries = []

    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            # Check if the line starts with ':::MLLOG'
            if line.startswith(":::MLLOG"):
                # Extract the JSON part of the line
                json_str = line[len(":::MLLOG "):].strip()
                try:
                    log_entry = json.loads(json_str)
                    log_entries.append(log_entry)
                except json.JSONDecodeError as e:
                    print(f"Failed to decode JSON: {e}")
                    continue

    # Get the directory of the log file
    output_dir = os.path.dirname(log_file_path)
    # Define the output JSON file path
    output_json_path = os.path.join(output_dir, 'mlperf_log.json')

    # Write the aggregated log entries to the output JSON file
    with open(output_json_path, 'w') as output_file:
        json.dump(log_entries, output_file, indent=4)

    print(f"Aggregated log entries written to {output_json_path}")

# Example usage
log_file_path = '/home/ubuntu/benchmarking-sd/g5.2xlarge/log/mlperf_log.txt'  # Replace with your actual log file path

aggregate_logs(log_file_path)

