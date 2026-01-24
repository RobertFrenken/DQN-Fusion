#!/usr/bin/env python3
"""Parameter budget assistant

Uses the MODEL_SIZE_CALCULATIONS rules to compute student/teacher/TA parameter targets
and invokes `scripts/hyperparam_search.py` (autotune) to find concrete model hyperparameters
matching those targets.

Outputs a concise JSON/CSV summary of proposed architectures.

Example:
  scripts/parameter_budget.py --onboard-max 175000 --M 3 --kappa 20 --student-model vgae_student --teacher-model vgae --precision fp32

"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from math import sqrt
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_hyperparam_search(model, target, top=1, autotune=True, random_iterations=300, refine_iterations=80, csv_path=None):
    cmd = [sys.executable, 'scripts/hyperparam_search.py', '--model', model, '--top', str(top), '--target', str(int(target))]
    if autotune:
        cmd += ['--autotune', '--random-iterations', str(random_iterations), '--refine-iterations', str(refine_iterations)]
    if csv_path:
        cmd += ['--csv', csv_path]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    # parse CSV
    import csv
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def compute_targets(onboard_max, M, kappa, precision='fp32'):
    # student per model
    N_s_per_model = onboard_max / M
    # teacher per model
    N_t_per_model = kappa * N_s_per_model
    use_ta = kappa > 10
    if use_ta:
        N_ta = int(sqrt(N_t_per_model * N_s_per_model))
    else:
        N_ta = None
    return int(N_s_per_model), int(N_t_per_model), N_ta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onboard-max', type=int, required=True)
    p.add_argument('--M', type=int, default=3)
    p.add_argument('--kappa', type=float, default=20.0)
    p.add_argument('--precision', choices=['fp32','int8'], default='fp32')
    p.add_argument('--student-model', choices=['vgae_student','gat_student','dqn_student'], required=True)
    p.add_argument('--teacher-model', choices=['vgae','gat','dqn'], required=True)
    p.add_argument('--top', type=int, default=1)
    p.add_argument('--workdir', type=str, default='experiment_results')
    p.add_argument('--random-iterations', type=int, default=400)
    p.add_argument('--refine-iterations', type=int, default=80)
    args = p.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    out = {
        'onboard_max': args.onboard_max,
        'M': args.M,
        'kappa': args.kappa,
        'precision': args.precision,
        'student_model': args.student_model,
        'teacher_model': args.teacher_model,
    }

    N_s, N_t, N_ta = compute_targets(args.onboard_max, args.M, args.kappa, args.precision)
    out['student_target'] = N_s
    out['teacher_target'] = N_t
    out['ta_target'] = N_ta

    # run student search
    with tempfile.NamedTemporaryFile(prefix='student_search_', suffix='.csv', delete=False) as tf:
        csv_student = tf.name
    print(f"Searching student hyperparameters to match ~{N_s} params...")
    student_rows = run_hyperparam_search(args.student_model, N_s, top=args.top, autotune=True, random_iterations=args.random_iterations, refine_iterations=args.refine_iterations, csv_path=csv_student)
    out['student_candidates'] = student_rows[:args.top]

    # teacher search
    with tempfile.NamedTemporaryFile(prefix='teacher_search_', suffix='.csv', delete=False) as tf:
        csv_teacher = tf.name
    print(f"Searching teacher hyperparameters to match ~{N_t} params...")
    teacher_rows = run_hyperparam_search(args.teacher_model, N_t, top=args.top, autotune=True, random_iterations=args.random_iterations, refine_iterations=args.refine_iterations, csv_path=csv_teacher)
    out['teacher_candidates'] = teacher_rows[:args.top]

    # TA if needed
    if N_ta is not None:
        with tempfile.NamedTemporaryFile(prefix='ta_search_', suffix='.csv', delete=False) as tf:
            csv_ta = tf.name
        print(f"Searching TA hyperparameters to match ~{N_ta} params...")
        ta_rows = run_hyperparam_search(args.teacher_model.replace('vgae','vgae_student') if 'vgae' in args.teacher_model else args.teacher_model, N_ta, top=args.top, autotune=True, random_iterations=args.random_iterations // 2, refine_iterations=args.refine_iterations // 2, csv_path=csv_ta)
        out['ta_candidates'] = ta_rows[:args.top]

    # save JSON summary
    out_path = Path(args.workdir) / f'budget_proposal_M{args.M}_k{int(args.kappa)}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('Wrote summary to', out_path)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
