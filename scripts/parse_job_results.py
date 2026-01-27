#!/usr/bin/env python3
"""
Parse SLURM job output files to extract timing and batch size information.
"""
import re
from pathlib import Path
from datetime import datetime
import json

def parse_timestamp(line):
    """Extract timestamp from SLURM log line."""
    # Format: "Start time: Mon Jan 27 04:40:07 EST 2025" or "Mon Jan 26 11:57:53 PM EST 2026"
    match = re.search(r':\s+(.+)$', line)
    if match:
        timestamp_str = match.group(1).strip()

        # Try different formats
        formats = [
            "%a %b %d %H:%M:%S %Z %Y",  # 24-hour format
            "%a %b %d %I:%M:%S %p %Z %Y",  # 12-hour format with AM/PM
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                return dt
            except ValueError:
                continue

        # If all formats fail, return None
        return None
    return None

def parse_duration(line):
    """Extract duration from line like 'Duration: 00:02:35'."""
    match = re.search(r'Duration:\s+(\d{2}:\d{2}:\d{2})', line)
    if match:
        return match.group(1)
    return None

def parse_batch_size(content, frozen_config_path=None):
    """Extract batch size from log content or frozen config."""
    # First, try to extract from log content
    patterns = [
        r'batch_size[\'"]?\s*[:=]\s*(\d+)',
        r'Batch size:\s*(\d+)',
        r'"batch_size":\s*(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return int(match.group(1))

    # If not found in logs, try to read from frozen config
    if frozen_config_path and Path(frozen_config_path).exists():
        try:
            with open(frozen_config_path, 'r') as f:
                config = json.load(f)

            # Try different config paths where batch_size might be
            batch_size_paths = [
                config.get('training', {}).get('batch_size'),
                config.get('training', {}).get('fusion_agent', {}).get('fusion_batch_size'),
                config.get('datamodule', {}).get('batch_size'),
            ]

            for bs in batch_size_paths:
                if bs is not None:
                    return int(bs)
        except Exception:
            pass

    return None

def parse_exit_code(content):
    """Extract exit code from log."""
    match = re.search(r'Exit code:\s*(\d+)', content)
    if match:
        return int(match.group(1))
    return None

def parse_slurm_output(file_path):
    """Parse a single SLURM output file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract information
        result = {
            'file': str(file_path),
            'start_time': None,
            'end_time': None,
            'duration': None,
            'batch_size': None,
            'exit_code': None,
            'status': 'UNKNOWN',
            'error': None,
            'frozen_config_path': None
        }

        # Parse timestamps and frozen config path
        for line in content.split('\n'):
            if 'Start time:' in line:
                result['start_time'] = parse_timestamp(line)
            elif 'End time:' in line:
                result['end_time'] = parse_timestamp(line)
            elif 'Duration:' in line:
                result['duration'] = parse_duration(line)
            elif 'Frozen Config:' in line:
                # Extract path from line like "Frozen Config: /path/to/config.json"
                match = re.search(r'Frozen Config:\s+(.+\.json)', line)
                if match:
                    result['frozen_config_path'] = match.group(1).strip()

        # Parse batch size (pass frozen config path if found)
        result['batch_size'] = parse_batch_size(content, result.get('frozen_config_path'))

        # Parse exit code
        result['exit_code'] = parse_exit_code(content)

        # Determine status - check for success/failure messages
        if '✅ JOB COMPLETED SUCCESSFULLY' in content:
            result['status'] = 'SUCCESS'
            result['exit_code'] = 0
        elif '❌ Training failed:' in content or '❌ Failed to load frozen config:' in content:
            result['status'] = 'FAILED'
            if result['exit_code'] is None:
                result['exit_code'] = 1
        elif result['exit_code'] == 0:
            result['status'] = 'SUCCESS'
        elif result['exit_code'] is not None and result['exit_code'] != 0:
            result['status'] = 'FAILED'

        # Check for specific error types
        if 'CUDA out of memory' in content or 'OOM' in content:
            result['error'] = 'OOM'
            result['status'] = 'FAILED'
        elif 'CUDA error' in content:
            result['error'] = 'CUDA_ERROR'
            result['status'] = 'FAILED'
        elif 'Cancelled' in content or 'DUE TO DEPENDENCY' in content:
            result['error'] = 'CANCELLED'
            result['status'] = 'CANCELLED'
        elif 'Training failed' in content and result['error'] is None:
            result['error'] = 'ERROR'
            result['status'] = 'FAILED'

        # Calculate duration if not in logs
        if result['duration'] is None and result['start_time'] and result['end_time']:
            duration_seconds = (result['end_time'] - result['start_time']).total_seconds()
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            seconds = int(duration_seconds % 60)
            result['duration'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        return result

    except Exception as e:
        return {
            'file': str(file_path),
            'error': f"Parse error: {str(e)}",
            'status': 'PARSE_ERROR'
        }

def extract_metadata_from_path(file_path):
    """Extract dataset, model, size, mode from canonical path."""
    parts = Path(file_path).parts

    # Find indices
    try:
        dataset_idx = parts.index('automotive') + 1
        learning_type_idx = dataset_idx + 1
        model_idx = learning_type_idx + 1
        size_idx = model_idx + 1
        distill_idx = size_idx + 1
        mode_idx = distill_idx + 1

        return {
            'dataset': parts[dataset_idx],
            'learning_type': parts[learning_type_idx],
            'model': parts[model_idx],
            'size': parts[size_idx],
            'distillation': parts[distill_idx],
            'mode': parts[mode_idx]
        }
    except (ValueError, IndexError):
        return {
            'dataset': 'UNKNOWN',
            'learning_type': 'UNKNOWN',
            'model': 'UNKNOWN',
            'size': 'UNKNOWN',
            'distillation': 'UNKNOWN',
            'mode': 'UNKNOWN'
        }

def main():
    # Find all .out files
    base_dir = Path('/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/experimentruns')
    out_files = list(base_dir.glob('**/*.out'))

    print(f"Found {len(out_files)} SLURM output files\n")

    results = []
    for out_file in sorted(out_files):
        print(f"Parsing: {out_file.name}")
        parsed = parse_slurm_output(out_file)
        metadata = extract_metadata_from_path(out_file)
        parsed.update(metadata)
        results.append(parsed)

    # Save to JSON
    output_file = Path('/users/PAS2022/rf15/CAN-Graph-Test/KD-GAT/job_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {output_file}")

    # Print summary table
    print("\n" + "="*120)
    print(f"{'Dataset':<10} {'Model':<6} {'Size':<8} {'Mode':<12} {'Status':<10} {'Duration':<10} {'Batch':<8} {'Error':<15}")
    print("="*120)

    for r in results:
        dataset = r.get('dataset', 'UNKNOWN') or 'UNKNOWN'
        model = r.get('model', 'UNKNOWN') or 'UNKNOWN'
        size = r.get('size', 'UNKNOWN') or 'UNKNOWN'
        mode = r.get('mode', 'UNKNOWN') or 'UNKNOWN'
        status = r.get('status', 'UNKNOWN') or 'UNKNOWN'
        duration = r.get('duration') or 'N/A'
        batch_size = r.get('batch_size')
        if batch_size is not None:
            batch_size = str(batch_size)
        else:
            batch_size = 'N/A'
        error = r.get('error') or ''

        print(f"{dataset:<10} {model:<6} {size:<8} {mode:<12} {status:<10} {duration:<10} {batch_size:<8} {error:<15}")

    print("="*120)

    # Summary stats
    total = len(results)
    success = sum(1 for r in results if r.get('status') == 'SUCCESS')
    failed = sum(1 for r in results if r.get('status') == 'FAILED')

    print(f"\nTotal: {total} jobs | Success: {success} | Failed: {failed}")

if __name__ == '__main__':
    main()
