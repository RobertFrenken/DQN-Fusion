#!/usr/bin/env python3
"""Hyperparameter grid search for model parameter budgets.

Features:
- Search over VGAE and StudentVGAE hyperparameters (hidden_dims, heads, embedding_dim)
- Instantiate models directly from model classes (GraphAutoencoderNeighborhood, GATWithJK, etc.)
- Compute total parameter count and detailed module breakdown
- Output top candidates by closeness to target parameter budget
- Save results to CSV for further analysis

Usage examples:
  scripts/hyperparam_search.py --model vgae --target 1740000 --grid-file /path/to/grid.json --top 10
  scripts/hyperparam_search.py --model vgae_student --hidden-dims "[200,100,24];[180,90,24]" --heads 2,4,8 --emb 8,16 --top 5

"""
import argparse
import csv
import json
import math
import os
import sys
from typing import List, Tuple

# Ensure project src is importable when script run from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.vgae import GraphAutoencoderNeighborhood
from src.models.models import GATWithJK, create_dqn_teacher, create_dqn_student
from src.config.hydra_zen_configs import VGAEConfig, StudentVGAEConfig, GATConfig, StudentGATConfig, DQNConfig, StudentDQNConfig
import random
import subprocess
import statistics


def count_params(module):
    return sum(int(p.numel()) for p in module.parameters() if p is not None)


def breakdown_vgae(model) -> dict:
    """Return a breakdown of parameter counts for a VGAE model instance."""
    breakdown = {}
    # id embedding
    if hasattr(model, 'id_embedding'):
        breakdown['id_embedding'] = count_params(model.id_embedding)
    # encoder (per-layer)
    enc = getattr(model, 'encoder_layers', None)
    if enc is not None:
        enc_params = 0
        for i, layer in enumerate(enc):
            key = f'encoder_layer_{i}'
            val = count_params(layer)
            breakdown[key] = val
            enc_params += val
        breakdown['encoder'] = enc_params
    # z heads
    z_params = 0
    for k in ['z_mean', 'z_logvar']:
        if hasattr(model, k):
            z_params += count_params(getattr(model, k))
    breakdown['z_heads'] = z_params
    # decoder (per-layer)
    dec = getattr(model, 'decoder_layers', None)
    if dec is not None:
        dec_params = 0
        for i, layer in enumerate(dec):
            key = f'decoder_layer_{i}'
            val = count_params(layer)
            breakdown[key] = val
            dec_params += val
        breakdown['decoder'] = dec_params
    # neighborhood decoder
    if hasattr(model, 'neighborhood_decoder'):
        # neighborhood_decoder is an nn.Sequential: compute per-submodule breakdown
        nd = model.neighborhood_decoder
        nd_total = 0
        for i, sub in enumerate(nd):
            key = f'neigh_layer_{i}'
            val = count_params(sub)
            breakdown[key] = val
            nd_total += val
        breakdown['neighborhood_decoder'] = nd_total
    # canid classifier
    if hasattr(model, 'canid_classifier'):
        breakdown['canid_classifier'] = count_params(model.canid_classifier)
    # rest
    known = set(k for k in breakdown.keys())
    total = count_params(model)
    known_sum = sum(breakdown.get(k, 0) for k in known)
    breakdown['other'] = max(0, total - known_sum)
    breakdown['total'] = total
    return breakdown


def evaluate_vgae_combo(hidden_dims: List[int], heads: int, emb: int, latent_dim: int = 48, num_ids=50, mlp_hidden: int = None):
    cfg = VGAEConfig()
    cfg.hidden_dims = list(hidden_dims)
    cfg.attention_heads = heads
    cfg.embedding_dim = emb
    cfg.latent_dim = latent_dim
    # Instantiate model directly
    model = GraphAutoencoderNeighborhood(
        num_ids=num_ids,
        in_channels=cfg.input_dim,
        hidden_dims=list(hidden_dims),
        latent_dim=latent_dim,
        encoder_heads=heads,
        decoder_heads=heads,
        embedding_dim=emb,
        dropout=cfg.dropout,
        batch_norm=getattr(cfg, 'batch_norm', True),
        mlp_hidden=mlp_hidden
    )
    total = count_params(model)
    breakdown = breakdown_vgae(model)
    return total, breakdown


def evaluate_vgae_student_combo(encoder_dims: List[int], heads: int, emb: int, latent_dim: int = 24, num_ids=50, mlp_hidden: int = None):
    cfg = StudentVGAEConfig()
    cfg.encoder_dims = list(encoder_dims)
    cfg.attention_heads = heads
    cfg.embedding_dim = emb
    cfg.latent_dim = latent_dim
    # Instantiate model directly
    model = GraphAutoencoderNeighborhood(
        num_ids=num_ids,
        in_channels=cfg.input_dim,
        hidden_dims=list(encoder_dims),
        latent_dim=latent_dim,
        encoder_heads=heads,
        decoder_heads=heads,
        embedding_dim=emb,
        dropout=cfg.dropout,
        batch_norm=getattr(cfg, 'batch_norm', True),
        mlp_hidden=mlp_hidden
    )
    total = count_params(model)
    breakdown = breakdown_vgae(model)
    return total, breakdown


# ---- GAT/DQN support and breakdowns ----

def breakdown_gat(model) -> dict:
    breakdown = {}
    if hasattr(model, 'id_embedding'):
        breakdown['id_embedding'] = count_params(model.id_embedding)
    if hasattr(model, 'convs'):
        breakdown['convs'] = sum(count_params(c) for c in model.convs)
    if hasattr(model, 'jk'):
        breakdown['jk'] = count_params(model.jk)
    if hasattr(model, 'fc_layers'):
        breakdown['fc_layers'] = sum(count_params(l) for l in model.fc_layers)
    if hasattr(model, 'fc_layers'):
        # guess final classifier is last Linear in fc_layers
        pass
    total = count_params(model)
    known_sum = sum(v for v in breakdown.values())
    breakdown['other'] = max(0, total - known_sum)
    breakdown['total'] = total
    return breakdown


def evaluate_gat_combo(hidden_channels: int, heads: int, num_layers: int, num_fc_layers: int, embedding_dim: int = 32, num_ids=50):
    cfg = GATConfig()
    cfg.hidden_channels = hidden_channels
    cfg.heads = heads
    cfg.num_layers = num_layers
    cfg.num_fc_layers = num_fc_layers
    cfg.embedding_dim = embedding_dim
    # Instantiate model directly
    model = GATWithJK(
        num_ids=num_ids,
        in_channels=cfg.input_dim,
        hidden_channels=hidden_channels,
        out_channels=cfg.output_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=cfg.dropout,
        num_fc_layers=num_fc_layers,
        embedding_dim=embedding_dim
    )
    total = count_params(model)
    breakdown = breakdown_gat(model)
    return total, breakdown


def evaluate_gat_student_combo(hidden_channels: int, heads: int, num_layers: int, num_fc_layers: int, embedding_dim: int = 8, num_ids=50):
    cfg = StudentGATConfig()
    cfg.hidden_channels = hidden_channels
    cfg.heads = heads
    cfg.num_layers = num_layers
    cfg.num_fc_layers = num_fc_layers
    cfg.embedding_dim = embedding_dim
    # Instantiate model directly
    model = GATWithJK(
        num_ids=num_ids,
        in_channels=cfg.input_dim,
        hidden_channels=hidden_channels,
        out_channels=cfg.output_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=cfg.dropout,
        num_fc_layers=num_fc_layers,
        embedding_dim=embedding_dim
    )
    total = count_params(model)
    breakdown = breakdown_gat(model)
    return total, breakdown


def breakdown_dqn(model) -> dict:
    breakdown = {}
    if hasattr(model, 'network'):
        breakdown['network'] = count_params(model.network)
    total = count_params(model)
    breakdown['other'] = max(0, total - breakdown.get('network', 0))
    breakdown['total'] = total
    return breakdown


def evaluate_dqn_combo(hidden_units: int, num_layers: int, input_dim: int = 20, output_dim: int = 11):
    cfg = DQNConfig()
    cfg.hidden_units = hidden_units
    cfg.num_layers = num_layers
    cfg.input_dim = input_dim
    cfg.output_dim = output_dim
    # Instantiate model directly
    model = create_dqn_teacher(cfg, num_ids=50)
    total = count_params(model)
    breakdown = breakdown_dqn(model)
    return total, breakdown


def evaluate_dqn_student_combo(hidden_units: int, num_layers: int, input_dim: int = 20, output_dim: int = 11):
    cfg = StudentDQNConfig()
    cfg.hidden_units = hidden_units
    cfg.num_layers = num_layers
    cfg.input_dim = input_dim
    cfg.output_dim = output_dim
    # Instantiate model directly
    model = create_dqn_student(cfg, num_ids=50)
    total = count_params(model)
    breakdown = breakdown_dqn(model)
    return total, breakdown


def parse_hidden_dims_list(s: str) -> List[List[int]]:
    # accepts formats: "[896,448,336,48];[1024,512,384,48]" or semicolon or pipe separated, or single JSON list
    parts = [p.strip() for p in s.split(';') if p.strip()]
    out = []
    for p in parts:
        try:
            arr = json.loads(p)
            if isinstance(arr, list):
                out.append([int(x) for x in arr])
                continue
        except Exception:
            pass
        # fallback split by comma
        nums = [int(x) for x in p.replace('[', '').replace(']', '').split(',') if x.strip()]
        out.append(nums)
    return out


def write_csv(path: str, rows: List[dict]):
    keys = []
    if rows:
        # collect union of keys
        ks = set()
        for r in rows:
            ks.update(r.keys())
        keys = sorted(ks)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['vgae', 'vgae_student', 'gat', 'gat_student', 'dqn', 'dqn_student'], required=True)
    p.add_argument('--hidden-dims', type=str, default='[896,448,336,48];[1024,512,384,48];[1152,576,432,48];[1536,768,576,48]')
    p.add_argument('--encoder-dims', type=str, default='[200,100,24];[180,90,24];[220,110,24]')
    p.add_argument('--heads', type=str, default='4,8,16')
    p.add_argument('--emb', type=str, default='32,64')
    p.add_argument('--latent', type=int, default=None)
    p.add_argument('--num-ids', type=int, default=50)
    p.add_argument('--target', type=int, default=None, help='Target parameter budget; if omitted uses config default')
    p.add_argument('--autotune', action='store_true', help='Run autotune (random search + local refinement)')
    p.add_argument('--random-iterations', type=int, default=200, help='Random search iterations for autotune')
    p.add_argument('--refine-iterations', type=int, default=40, help='Local refinement iterations for autotune')
    p.add_argument('--seed', type=int, default=42, help='Random seed for autotune')
    p.add_argument('--top', type=int, default=10, help='Number of top candidates to print')
    p.add_argument('--csv', type=str, default=None, help='Path to write CSV of all candidates')
    args = p.parse_args()

    heads = [int(x) for x in args.heads.split(',') if x.strip()]
    emb = [int(x) for x in args.emb.split(',') if x.strip()]

    rows = []

    if args.model == 'vgae':
        hidden_list = parse_hidden_dims_list(args.hidden_dims)
        for hd in hidden_list:
            for h in heads:
                for e in emb:
                    total, breakdown = evaluate_vgae_combo(hd, h, e, latent_dim=(args.latent or 48), num_ids=args.num_ids)
                    row = {
                        'model': 'vgae',
                        'hidden_dims': str(hd),
                        'heads': h,
                        'embedding': e,
                        'total_params': total,
                    }
                    row.update({f'part_{k}': v for k, v in breakdown.items()})
                    rows.append(row)
    elif args.model == 'vgae_student':
        encoder_list = parse_hidden_dims_list(args.encoder_dims)
        for hd in encoder_list:
            for h in heads:
                for e in emb:
                    total, breakdown = evaluate_vgae_student_combo(hd, h, e, latent_dim=(args.latent or 24), num_ids=args.num_ids)
                    row = {
                        'model': 'vgae_student',
                        'encoder_dims': str(hd),
                        'heads': h,
                        'embedding': e,
                        'total_params': total,
                    }
                    row.update({f'part_{k}': v for k, v in breakdown.items()})
                    rows.append(row)
    elif args.model in ('gat', 'gat_student'):
        # parse simple ranges from args.hidden_dims (single integer for hidden_channels)
        hidden_vals = [int(x) for x in args.hidden_dims.split(',') if x.strip()]
        num_layers_list = [int(x) for x in args.encoder_dims.split(',') if x.strip()]
        num_fc_list = [1,2,3]
        for hidden in hidden_vals:
            for nl in num_layers_list:
                for h in heads:
                    for fc in num_fc_list:
                        if args.model == 'gat':
                            total, breakdown = evaluate_gat_combo(hidden, h, nl, fc, embedding_dim=(emb[0] if len(emb)==1 else emb[0]), num_ids=args.num_ids)
                            row = {'model': 'gat', 'hidden_channels': hidden, 'num_layers': nl, 'heads': h, 'num_fc_layers': fc, 'total_params': total}
                        else:
                            total, breakdown = evaluate_gat_student_combo(hidden, h, nl, fc, embedding_dim=(emb[0] if len(emb)==1 else emb[0]), num_ids=args.num_ids)
                            row = {'model': 'gat_student', 'hidden_channels': hidden, 'num_layers': nl, 'heads': h, 'num_fc_layers': fc, 'total_params': total}
                        row.update({f'part_{k}': v for k, v in breakdown.items()})
                        rows.append(row)
    elif args.model in ('dqn', 'dqn_student'):
        hidden_vals = [int(x) for x in args.hidden_dims.split(',') if x.strip()]
        num_layers_list = [int(x) for x in args.encoder_dims.split(',') if x.strip()]
        for hidden in hidden_vals:
            for nl in num_layers_list:
                total, breakdown = (evaluate_dqn_combo(hidden, nl) if args.model=='dqn' else evaluate_dqn_student_combo(hidden, nl))
                row = {'model': args.model, 'hidden_units': hidden, 'num_layers': nl, 'total_params': total}
                row.update({f'part_{k}': v for k, v in breakdown.items()})
                rows.append(row)

    # determine target
    if args.target is None:
        # use config default
        if args.model == 'vgae':
            target = VGAEConfig().target_parameters
        else:
            target = StudentVGAEConfig().target_parameters
    else:
        target = args.target

    rows_sorted = sorted(rows, key=lambda r: abs(r['total_params'] - target))

    def _print_rows(rows_to_print):
        print(f"Top {args.top} candidates for model={args.model} target={target}")
        for r in rows_to_print[:args.top]:
            pct = (r['total_params'] / target) * 100 if target > 0 else float('inf')
            parts = ','.join(f"{k}:{v}" for k, v in r.items() if k.startswith('part_'))
            meta = r.get('hidden_dims') or r.get('encoder_dims') or r.get('hidden_channels') or r.get('hidden_units') or ''
            print(f"{meta} params={r['total_params']} ({pct:.2f}%) breakdown={{" + parts + "}}")

    if args.autotune:
        # simple random search + local refinement
        random.seed(args.seed)
        best = None
        best_row = None
        # random search over current rows parameterizations if rows exist (grid mode)
        if rows:
            for i in range(min(len(rows), args.random_iterations)):
                r = random.choice(rows)
                if best is None or abs(r['total_params'] - target) < abs(best - target):
                    best = r['total_params']
                    best_row = r
        # if no rows or random_explore, do random generation for specific models
        def _random_candidate():
            if args.model == 'vgae':
                # sample small multipliers of base template
                scales = [0.75, 1.0, 1.5, 2.0, 3.0]
                hd = random.choice(parse_hidden_dims_list(args.hidden_dims))
                # randomly scale
                sc = random.choice(scales)
                hd = [max(8, int(x*sc)) for x in hd]
                h = random.choice(heads)
                e = random.choice(emb)
                total, breakdown = evaluate_vgae_combo(hd, h, e, latent_dim=(args.latent or 48), num_ids=args.num_ids)
                row = {'model': 'vgae', 'hidden_dims': str(hd), 'heads': h, 'embedding': e, 'total_params': total}
                row.update({f'part_{k}': v for k, v in breakdown.items()})
                return row
            if args.model == 'vgae_student':
                hd = random.choice(parse_hidden_dims_list(args.encoder_dims))
                sc = random.choice([0.75,1.0,1.25,1.5])
                hd = [max(8,int(x*sc)) for x in hd]
                h = random.choice(heads)
                e = random.choice(emb)
                total, breakdown = evaluate_vgae_student_combo(hd, h, e, latent_dim=(args.latent or 24), num_ids=args.num_ids)
                row = {'model': 'vgae_student', 'encoder_dims': str(hd), 'heads': h, 'embedding': e, 'total_params': total}
                row.update({f'part_{k}': v for k, v in breakdown.items()})
                return row
            if args.model in ('gat','gat_student'):
                hidden = random.choice([int(x) for x in args.hidden_dims.split(',') if x.strip()])
                nl = random.choice([int(x) for x in args.encoder_dims.split(',') if x.strip()])
                h = random.choice(heads)
                fc = random.choice([1,2,3])
                if args.model == 'gat':
                    total, breakdown = evaluate_gat_combo(hidden, h, nl, fc, embedding_dim=(emb[0] if len(emb)==1 else emb[0]), num_ids=args.num_ids)
                    row = {'model': 'gat', 'hidden_channels': hidden, 'num_layers': nl, 'heads': h, 'num_fc_layers': fc, 'total_params': total}
                else:
                    total, breakdown = evaluate_gat_student_combo(hidden, h, nl, fc, embedding_dim=(emb[0] if len(emb)==1 else emb[0]), num_ids=args.num_ids)
                    row = {'model': 'gat_student', 'hidden_channels': hidden, 'num_layers': nl, 'heads': h, 'num_fc_layers': fc, 'total_params': total}
                row.update({f'part_{k}': v for k, v in breakdown.items()})
                return row
            if args.model in ('dqn','dqn_student'):
                hidden = random.choice([int(x) for x in args.hidden_dims.split(',') if x.strip()])
                nl = random.choice([int(x) for x in args.encoder_dims.split(',') if x.strip()])
                total, breakdown = (evaluate_dqn_combo(hidden, nl) if args.model=='dqn' else evaluate_dqn_student_combo(hidden, nl))
                row = {'model': args.model, 'hidden_units': hidden, 'num_layers': nl, 'total_params': total}
                row.update({f'part_{k}': v for k, v in breakdown.items()})
                return row
            return None

        # random search
        rand_rows = []
        for i in range(args.random_iterations):
            r = _random_candidate()
            rand_rows.append(r)
        # take best random
        if rand_rows:
            br = min(rand_rows, key=lambda x: abs(x['total_params']-target))
            if best_row is None or abs(br['total_params']-target) < abs(best_row['total_params']-target):
                best_row = br
        # local refinement: perturb best_row
        if best_row is not None:
            refined = [best_row]
            for i in range(args.refine_iterations):
                base = random.choice(refined)
                # perturb depending on model
                if base['model']=='vgae':
                    hd = json.loads(base['hidden_dims'].replace("'", '"')) if isinstance(base['hidden_dims'], str) else base['hidden_dims']
                    # perturb one element
                    idx = random.randrange(len(hd))
                    hd2 = hd.copy()
                    hd2[idx] = max(8, int(hd2[idx] * random.uniform(0.8,1.25)))
                    r = evaluate_vgae_combo(hd2, base['heads'], base['embedding'], latent_dim=(args.latent or 48), num_ids=args.num_ids)
                    row = {'model':'vgae', 'hidden_dims':str(hd2), 'heads':base['heads'], 'embedding':base['embedding'], 'total_params':r[0]}
                    row.update({f'part_{k}': v for k, v in r[1].items()})
                elif base['model']=='vgae_student':
                    hd = json.loads(base['encoder_dims'].replace("'", '"')) if isinstance(base.get('encoder_dims'), str) else base.get('encoder_dims')
                    idx = random.randrange(len(hd))
                    hd2 = hd.copy(); hd2[idx] = max(8,int(hd2[idx]*random.uniform(0.8,1.25)))
                    r = evaluate_vgae_student_combo(hd2, base['heads'], base['embedding'], latent_dim=(args.latent or 24), num_ids=args.num_ids)
                    row = {'model':'vgae_student','encoder_dims':str(hd2),'heads':base['heads'],'embedding':base['embedding'],'total_params':r[0]}
                    row.update({f'part_{k}': v for k, v in r[1].items()})
                else:
                    # simple perturb for GAT/DQN
                    if base['model'] in ('gat','gat_student'):
                        hidden = int(base['hidden_channels'] * random.uniform(0.8,1.25))
                        nl = int(base['num_layers'])
                        h = base['heads']
                        fc = int(base['num_fc_layers'])
                        if base['model']=='gat':
                            r = evaluate_gat_combo(hidden, h, nl, fc, embedding_dim=(emb[0] if len(emb)==1 else emb[0]), num_ids=args.num_ids)
                            row = {'model':'gat','hidden_channels':hidden,'num_layers':nl,'heads':h,'num_fc_layers':fc,'total_params':r[0]}
                            row.update({f'part_{k}': v for k, v in r[1].items()})
                        else:
                            r = evaluate_gat_student_combo(hidden, h, nl, fc, embedding_dim=(emb[0] if len(emb)==1 else emb[0]), num_ids=args.num_ids)
                            row = {'model':'gat_student','hidden_channels':hidden,'num_layers':nl,'heads':h,'num_fc_layers':fc,'total_params':r[0]}
                            row.update({f'part_{k}': v for k, v in r[1].items()})
                    else:
                        hidden = int(base['hidden_units'] * random.uniform(0.8,1.25))
                        nl = int(base['num_layers'])
                        r = (evaluate_dqn_combo(hidden,nl) if base['model']=='dqn' else evaluate_dqn_student_combo(hidden,nl))
                        row = {'model':base['model'],'hidden_units':hidden,'num_layers':nl,'total_params':r[0]}
                        row.update({f'part_{k}': v for k, v in r[1].items()})
                refined.append(row)
            # choose best refined
            br2 = min(refined, key=lambda x: abs(x['total_params']-target))
            rows_sorted = sorted(rows + refined + rand_rows, key=lambda r: abs(r['total_params'] - target))
            _print_rows(rows_sorted)
            if args.csv:
                write_csv(args.csv, rows_sorted)
                print(f"Wrote results to {args.csv}")
            return rows_sorted

    _print_rows(rows_sorted)

    if args.csv:
        write_csv(args.csv, rows_sorted)
        print(f"Wrote results to {args.csv}")

    return rows_sorted


if __name__ == '__main__':
    main()
