#!/usr/bin/env python3
"""
Create pipeline-level summary from job results.
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

def parse_duration(duration_str):
    """Convert HH:MM:SS to timedelta."""
    if duration_str == 'N/A' or not duration_str:
        return None
    parts = duration_str.split(':')
    return timedelta(hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2]))

def format_duration(td):
    """Format timedelta as HH:MM:SS."""
    if td is None:
        return 'N/A'
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def main():
    # Load job results
    results_file = Path('/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/job_results.json')
    with open(results_file, 'r') as f:
        jobs = json.load(f)

    # Group by pipeline (dataset + size)
    pipelines = defaultdict(lambda: {
        'jobs': [],
        'total_duration': timedelta(),
        'status': 'SUCCESS',
        'batch_sizes': set()
    })

    for job in jobs:
        dataset = job.get('dataset', 'UNKNOWN')
        size = job.get('size', 'UNKNOWN')
        pipeline_key = f"{dataset}_{size}"

        pipelines[pipeline_key]['jobs'].append(job)

        # Add duration
        duration_str = job.get('duration')
        if duration_str and duration_str != 'N/A':
            duration = parse_duration(duration_str)
            if duration:
                pipelines[pipeline_key]['total_duration'] += duration

        # Track batch sizes
        batch_size = job.get('batch_size')
        if batch_size:
            pipelines[pipeline_key]['batch_sizes'].add(batch_size)

        # Update pipeline status
        if job.get('status') == 'FAILED':
            pipelines[pipeline_key]['status'] = 'FAILED'

    # Print pipeline summary
    print("\n" + "="*140)
    print("PIPELINE SUMMARY (3-Stage: VGAE → GAT → DQN)")
    print("="*140)
    print(f"{'Pipeline':<20} {'Size':<8} {'Status':<10} {'Total Duration':<15} {'Batch Sizes':<15} {'Jobs':<10} {'Details':<40}")
    print("="*140)

    for pipeline_key in sorted(pipelines.keys()):
        data = pipelines[pipeline_key]
        dataset, size = pipeline_key.rsplit('_', 1)

        total_duration = format_duration(data['total_duration'])
        batch_sizes = ', '.join(str(bs) for bs in sorted(data['batch_sizes']))
        num_jobs = len(data['jobs'])
        status = data['status']

        # Count successes and failures
        successes = sum(1 for j in data['jobs'] if j.get('status') == 'SUCCESS')
        failures = sum(1 for j in data['jobs'] if j.get('status') == 'FAILED')

        details = f"{successes} success, {failures} failed"

        print(f"{dataset:<20} {size:<8} {status:<10} {total_duration:<15} {batch_sizes:<15} {num_jobs:<10} {details:<40}")

    print("="*140)

    # Print detailed job-level results
    print("\n" + "="*140)
    print("DETAILED JOB RESULTS")
    print("="*140)
    print(f"{'Dataset':<12} {'Model':<6} {'Size':<8} {'Mode':<12} {'Status':<10} {'Duration':<12} {'Batch':<8} {'Error':<15}")
    print("="*140)

    for job in sorted(jobs, key=lambda x: (x.get('dataset', ''), x.get('size', ''), x.get('model', ''))):
        dataset = job.get('dataset') or 'UNKNOWN'
        model = job.get('model') or 'UNKNOWN'
        size = job.get('size') or 'UNKNOWN'
        mode = job.get('mode') or 'UNKNOWN'
        status = job.get('status') or 'UNKNOWN'
        duration = job.get('duration') or 'N/A'
        batch_size = job.get('batch_size')
        if batch_size is not None:
            batch_size = str(batch_size)
        else:
            batch_size = 'N/A'
        error = job.get('error') or ''

        print(f"{dataset:<12} {model:<6} {size:<8} {mode:<12} {status:<10} {duration:<12} {batch_size:<8} {error:<15}")

    print("="*140)

    # Summary statistics
    total_jobs = len(jobs)
    total_success = sum(1 for j in jobs if j.get('status') == 'SUCCESS')
    total_failed = sum(1 for j in jobs if j.get('status') == 'FAILED')

    # Total compute time
    total_compute = timedelta()
    for job in jobs:
        duration_str = job.get('duration')
        if duration_str and duration_str != 'N/A':
            duration = parse_duration(duration_str)
            if duration:
                total_compute += duration

    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total Jobs: {total_jobs}")
    print(f"   Successful: {total_success} ({total_success/total_jobs*100:.1f}%)")
    print(f"   Failed: {total_failed} ({total_failed/total_jobs*100:.1f}%)")
    print(f"   Total Compute Time: {format_duration(total_compute)}")

    # Breakdown by failure type
    failure_types = defaultdict(int)
    for job in jobs:
        if job.get('status') == 'FAILED':
            error_type = job.get('error', 'UNKNOWN')
            failure_types[error_type] += 1

    if failure_types:
        print(f"\n❌ FAILURE BREAKDOWN")
        for error_type, count in sorted(failure_types.items()):
            print(f"   {error_type}: {count}")

if __name__ == '__main__':
    main()
