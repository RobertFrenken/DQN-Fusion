#!/usr/bin/env python3
import subprocess
import sys

try:
    # Get full squeue output
    result = subprocess.run(
        ['squeue', '-u', 'rf15', '-o', '%.18i|%.100j|%.8T|%.20S'],
        capture_output=True,
        text=True
    )

    print("All jobs from squeue:")
    print("=" * 80)
    for line in result.stdout.split('\n'):
        if line.strip():
            print(line)

    print("\n" + "=" * 80)
    print("\nFiltering for teacher DQN jobs:")
    print("=" * 80)

    for line in result.stdout.split('\n'):
        if 'dqn' in line.lower() and ('teacher' in line.lower() or line.startswith(' ')):
            parts = line.split('|')
            if len(parts) >= 2:
                job_id = parts[0].strip()
                job_name = parts[1].strip() if len(parts) > 1 else ""
                if job_id and job_id != 'JOBID':
                    print(f"{job_id}: {job_name}")

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
