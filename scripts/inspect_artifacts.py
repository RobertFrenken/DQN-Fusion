"""
Inspect saved model artifacts and confirm they're state_dicts or contain expected keys.
Usage: conda run -n gnn-experiments python scripts/inspect_artifacts.py
"""
from pathlib import Path
import torch

artifacts = {
    'autoencoder': Path('experimentruns_test/automotive/hcrl_ch/unsupervised/vgae/teacher/no_distillation/autoencoder/vgae_autoencoder.pth'),
    'classifier': Path('experimentruns_test/automotive/hcrl_ch/supervised/gat/teacher/no_distillation/normal/gat_hcrl_ch_normal.pth'),
    'fusion_agent': Path('experimentruns_test/automotive/hcrl_ch/rl_fusion/vgae/student/no_distillation/fusion/models/fusion_agent_hcrl_ch.pth')
}

for name, p in artifacts.items():
    print(f"--- {name} -> {p}")
    if not p.exists():
        print('MISSING')
        continue
    try:
        ck = torch.load(p, map_location='cpu')
        print('loaded type:', type(ck))
        if isinstance(ck, dict):
            print('keys:', list(ck.keys()))
            # heuristics
            if all(isinstance(v, dict) for v in ck.values()):
                print('Looks like state-dict or nested dicts')
        else:
            print('Not a dict; examine repr')
            print(repr(ck)[:200])
    except Exception as e:
        print('ERROR loading:', e)
