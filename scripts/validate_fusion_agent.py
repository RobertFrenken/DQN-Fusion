"""
Load cached predictions and the fusion agent, run a quick validation pass and print metrics.
Usage: conda run -n gnn-experiments python scripts/validate_fusion_agent.py
"""
import sys
from pathlib import Path
# Ensure project root is importable
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import pickle
from src.models.dqn import EnhancedDQNFusionAgent

cache_train = Path('cache/fusion/hcrl_ch_train_predictions.pkl')
cache_val = Path('cache/fusion/hcrl_ch_val_predictions.pkl')
agents = Path('experimentruns_test/automotive/hcrl_ch/rl_fusion/vgae/student/no_distillation/fusion/models')
agent_f = agents / 'fusion_agent_hcrl_ch.pth'

print('loading caches...')
with open(cache_train,'rb') as f:
    t = pickle.load(f)
with open(cache_val,'rb') as f:
    v = pickle.load(f)
print('train sizes:', len(t['anomaly_scores']), len(t['gat_probs']), len(t['labels']))
print('val sizes:', len(v['anomaly_scores']), len(v['gat_probs']), len(v['labels']))

import torch
print('loading agent...')
# Try to inspect checkpoint and configure agent state_dim correctly
agent = None
if agent_f.exists():
    ck = None
    try:
        ck = torch.load(str(agent_f), map_location='cpu', weights_only=False)
    except Exception as e:
        print('weights_only=False load failed, will attempt agent.load_agent fallback:', e)
    if ck is None:
        try:
            temp_agent = EnhancedDQNFusionAgent(alpha_steps=21, device='cpu', state_dim=4)
            temp_agent.load_agent(str(agent_f))
            agent = temp_agent
            print('Loaded agent via agent.load_agent() fallback')
        except Exception as e2:
            print('agent.load_agent also failed:', e2)
    else:
        print('loaded agent checkpoint type:', type(ck))
        if isinstance(ck, dict) and 'q_network_state_dict' in ck:
            # infer in_features from first Linear weight
            qsd = ck['q_network_state_dict']
            # find first linear weight key
            weight_key = next((k for k in qsd.keys() if k.endswith('.weight')), None)
            if weight_key:
                in_features = qsd[weight_key].shape[1]
                print('inferred state_dim from q_network first weight:', in_features)
                agent = EnhancedDQNFusionAgent(alpha_steps=21, device='cpu', state_dim=in_features)
                agent.q_network.load_state_dict(qsd)
                if 'target_network_state_dict' in ck:
                    agent.target_network.load_state_dict(ck['target_network_state_dict'])
                print('Instantiated agent with compatible state_dim and loaded weights')
            else:
                print('No weight key found in q_network_state_dict; instantiating default agent')
                agent = EnhancedDQNFusionAgent(alpha_steps=21, device='cpu', state_dim=2)
        else:
            print('ck not a dict with q_network_state_dict; ck keys:', list(ck.keys()) if isinstance(ck, dict) else None)
else:
    print('agent file missing:', agent_f)

if agent is None:
    print('Failed to create/load agent; aborting validation')
    raise SystemExit(1)
else:
    print('agent ready: state_dim=', agent.state_dim)

    # Attempt to save a sanitized copy of the loaded agent so artifacts are future-safe
    try:
        sanitized_f = agent_f.with_name(agent_f.stem + '_sanitized.pth')
        agent.save_agent(str(sanitized_f))
        print('Sanitized agent saved to', sanitized_f)
    except Exception as e:
        print('Failed to save sanitized agent:', e)

# Normalize caches to same min length
min_len_train = min(len(t['anomaly_scores']), len(t['gat_probs']), len(t['labels']))
min_len_val = min(len(v['anomaly_scores']), len(v['gat_probs']), len(v['labels']))
if min_len_train != len(t['anomaly_scores']) or min_len_train != len(t['gat_probs']) or min_len_train != len(t['labels']):
    print('Trimming train cache to', min_len_train)
    t['anomaly_scores'] = t['anomaly_scores'][:min_len_train]
    t['gat_probs'] = t['gat_probs'][:min_len_train]
    t['labels'] = t['labels'][:min_len_train]
if min_len_val != len(v['anomaly_scores']) or min_len_val != len(v['gat_probs']) or min_len_val != len(v['labels']):
    print('Trimming val cache to', min_len_val)
    v['anomaly_scores'] = v['anomaly_scores'][:min_len_val]
    v['gat_probs'] = v['gat_probs'][:min_len_val]
    v['labels'] = v['labels'][:min_len_val]

# Prepare validation tuples
val_data = list(zip(v['anomaly_scores'], v['gat_probs'], v['labels']))
print('running manual validation on val cache... (n=', len(val_data),')')

import numpy as np
correct = 0
rewards = []
alphas = []
for anomaly, gat_prob, label in val_data:
    # Build 2-d state for q_network since the saved model appears to expect 2-d inputs
    state = np.array([anomaly, gat_prob], dtype=np.float32)
    import torch as _t
    s = _t.tensor(state, dtype=_t.float32).unsqueeze(0)
    with _t.no_grad():
        qv = agent.q_network(s)
        action_idx = int(qv.argmax(dim=1).item())
    alpha = agent.alpha_values[action_idx]
    fused = (1 - alpha) * anomaly + alpha * gat_prob
    pred = 1 if fused > 0.5 else 0
    correct += (pred == int(label))
    # compute reward using agent's function
    r = agent.compute_fusion_reward(pred, int(label), anomaly, gat_prob, alpha)
    rewards.append(r)
    alphas.append(alpha)

res = {
    'accuracy': correct / len(val_data) if len(val_data) else None,
    'avg_reward': sum(rewards) / len(rewards) if rewards else None,
    'avg_alpha': sum(alphas) / len(alphas) if alphas else None,
}
print('manual validation results:', res)
