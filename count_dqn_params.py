#!/usr/bin/env python3
"""
Quick script to count DQN model parameters
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from models.adaptive_fusion import QNetwork, EnhancedDQNFusionAgent

def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_number(num):
    """Format number with commas for readability"""
    return f"{num:,}"

# Initialize with default parameters
state_dim = 4  # anomaly_score, gat_prob, disagreement, avg_confidence  
alpha_steps = 21  # Default number of alpha values
hidden_dim = 128  # Default hidden dimension

print("=== DQN Model Parameter Analysis ===\n")

# Create single Q-network
q_net = QNetwork(state_dim=state_dim, action_dim=alpha_steps, hidden_dim=hidden_dim)
single_net_params = count_parameters(q_net)

print(f"📊 Network Architecture:")
print(f"   State dimension: {state_dim}")
print(f"   Action dimension: {alpha_steps} (alpha values: 0.0 to 1.0)")
print(f"   Hidden dimension: {hidden_dim}")
print()

print(f"🔢 Parameter Counts:")
print(f"   Single Q-Network: {format_number(single_net_params)} parameters")

# The EnhancedDQNFusionAgent has TWO networks: main and target
total_params = single_net_params * 2
print(f"   Complete DQN Agent: {format_number(total_params)} parameters")
print(f"   (Main Q-Network + Target Q-Network)")
print()

# Break down by layer
print(f"📋 Layer-by-layer breakdown:")
for name, param in q_net.named_parameters():
    param_count = param.numel()
    shape = list(param.shape)
    print(f"   {name:25} {str(shape):15} {format_number(param_count):>8} params")

print()

# Memory estimation
bytes_per_param = 4  # float32
single_net_memory = single_net_params * bytes_per_param
total_memory = total_params * bytes_per_param

print(f"💾 Memory Usage (float32):")
print(f"   Single Q-Network: {single_net_memory/1024:.1f} KB ({single_net_memory/1024/1024:.3f} MB)")
print(f"   Complete DQN Agent: {total_memory/1024:.1f} KB ({total_memory/1024/1024:.3f} MB)")
print()

# Model size categories
if total_params < 1000:
    size_category = "Tiny"
elif total_params < 10000:
    size_category = "Small"  
elif total_params < 100000:
    size_category = "Medium"
elif total_params < 1000000:
    size_category = "Large"
else:
    size_category = "Very Large"

print(f"📏 Model Size: {size_category}")
print(f"🚀 This is a very lightweight model suitable for fast inference and training!")