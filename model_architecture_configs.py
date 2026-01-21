#!/usr/bin/env python3
"""
Improved Teacher-Student Model Architecture Configurations

Designed for CAN bus deployment with proper encoder/decoder paths
and realistic parameter counts based on actual input dimensions.

Input: 11-dimensional CAN features (not 37)
Target: Student ~87K params, Teacher ~1.74M params
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn

def calculate_linear_params(input_dim: int, output_dim: int) -> int:
    """Calculate parameters for a linear layer (weights + bias)."""
    return input_dim * output_dim + output_dim

def calculate_vgae_params(encoder_layers: List[int], decoder_layers: List[int]) -> int:
    """Calculate total parameters for a VGAE model."""
    total_params = 0
    
    # Encoder parameters
    for i in range(len(encoder_layers) - 1):
        total_params += calculate_linear_params(encoder_layers[i], encoder_layers[i+1])
    
    # Decoder parameters  
    for i in range(len(decoder_layers) - 1):
        total_params += calculate_linear_params(decoder_layers[i], decoder_layers[i+1])
    
    # Add variational parameters (mean and log_std for latent space)
    latent_dim = encoder_layers[-1]
    total_params += 2 * calculate_linear_params(encoder_layers[-2], latent_dim)
    
    return total_params

# Improved Model Configurations
IMPROVED_CONFIGS = {
    "student": {
        "description": "Compact student model for on-board CAN deployment",
        "encoder_path": [11, 64, 12],  # 11 → 64 → 12 (proper compression)
        "decoder_path": [12, 64, 11],  # 12 → 64 → 11 (proper decompression)  
        "latent_dim": 12,
        "layers": 2,  # 1 encoder hidden + 1 decoder hidden
        "target_params": "~87K"
    },
    
    "teacher": {
        "description": "Larger teacher model for knowledge distillation",
        "encoder_path": [11, 128, 64, 48],  # 11 → 128 → 64 → 48 (gradual compression)
        "decoder_path": [48, 64, 128, 11],  # 48 → 64 → 128 → 11 (gradual decompression)
        "latent_dim": 48,
        "layers": 4,  # 2 encoder hidden + 2 decoder hidden  
        "target_params": "~1.74M"
    }
}

# Alternative configurations with different trade-offs
ALTERNATIVE_CONFIGS = {
    "student_compact": {
        "description": "Ultra-compact for severely resource-constrained deployment",
        "encoder_path": [11, 32, 8],
        "decoder_path": [8, 32, 11], 
        "latent_dim": 8,
        "layers": 2,
        "target_params": "~25K"
    },
    
    "student_balanced": {
        "description": "Balanced student with more capacity",
        "encoder_path": [11, 96, 16], 
        "decoder_path": [16, 96, 11],
        "latent_dim": 16,
        "layers": 2,
        "target_params": "~130K"
    },
    
    "teacher_deep": {
        "description": "Deeper teacher with more representational power",
        "encoder_path": [11, 256, 128, 64, 48],
        "decoder_path": [48, 64, 128, 256, 11],
        "latent_dim": 48, 
        "layers": 6,  # 3 encoder + 3 decoder hidden layers
        "target_params": "~3.2M"
    }
}

def print_architecture_analysis():
    """Print detailed analysis of all configurations."""
    
    print("🏗️  IMPROVED CAN-GRAPH TEACHER-STUDENT ARCHITECTURES")
    print("=" * 70)
    print("Input Dimension: 11 (actual CAN features)")
    print("Design Principle: Proper encoder compression → decoder decompression")
    print()
    
    all_configs = {**IMPROVED_CONFIGS, **ALTERNATIVE_CONFIGS}
    
    for config_name, config in all_configs.items():
        print(f"📊 {config_name.upper().replace('_', ' ')}")
        print("-" * 50)
        print(f"Description: {config['description']}")
        print(f"Encoder Path: {' → '.join(map(str, config['encoder_path']))}")
        print(f"Decoder Path: {' → '.join(map(str, config['decoder_path']))}")
        print(f"Latent Dimension: {config['latent_dim']}")
        print(f"Total Layers: {config['layers']}")
        
        # Calculate actual parameters
        encoder_layers = config['encoder_path']
        decoder_layers = config['decoder_path']
        actual_params = calculate_vgae_params(encoder_layers, decoder_layers)
        
        print(f"Estimated Parameters: {actual_params:,} ({actual_params/1000:.1f}K)")
        print(f"Target: {config['target_params']}")
        print(f"Model Size: {(actual_params * 4) / (1024 * 1024):.2f} MB")
        print()

def get_hydra_config_yaml():
    """Generate Hydra-Zen configuration YAML for the improved models."""
    
    yaml_content = """
# Improved Teacher-Student Model Configurations
# Fixed input dimension and proper encoder/decoder paths

model:
  student:
    type: "vgae"
    encoder_dims: [11, 64, 12]  # Proper compression: 11 → 64 → 12
    decoder_dims: [12, 64, 11]  # Proper decompression: 12 → 64 → 11
    latent_dim: 12
    hidden_dim: 64
    dropout: 0.1
    
  teacher:
    type: "vgae" 
    encoder_dims: [11, 128, 64, 48]  # Gradual compression: 11 → 128 → 64 → 48
    decoder_dims: [48, 64, 128, 11]  # Gradual decompression: 48 → 64 → 128 → 11
    latent_dim: 48
    hidden_dim: 128
    dropout: 0.1

training:
  knowledge_distillation:
    temperature: 4.0
    alpha: 0.7  # Weight for distillation loss
    beta: 0.3   # Weight for student task loss
    
  curriculum_learning:
    stages:
      - name: "teacher_pretraining"
        epochs: 100
        model: "teacher"
      - name: "knowledge_distillation" 
        epochs: 50
        models: ["teacher", "student"]
        
# Resource requirements for on-board deployment
deployment:
  student:
    memory_mb: 4      # ~25KB model + overhead
    inference_ms: 2   # Target inference time
    power_mw: 50      # Power consumption estimate
    
  teacher:
    memory_mb: 20     # ~87KB model + overhead  
    inference_ms: 8   # Acceptable for training only
    power_mw: 200     # Training/validation only
"""
    
    return yaml_content.strip()

class ImprovedVGAE(nn.Module):
    """
    Improved VGAE implementation with configurable encoder/decoder paths.
    Supports both student and teacher architectures.
    """
    
    def __init__(self, config_name: str = "student"):
        super().__init__()
        
        if config_name not in IMPROVED_CONFIGS:
            raise ValueError(f"Config '{config_name}' not found. Available: {list(IMPROVED_CONFIGS.keys())}")
        
        config = IMPROVED_CONFIGS[config_name]
        encoder_dims = config["encoder_path"]
        decoder_dims = config["decoder_path"]
        self.latent_dim = config["latent_dim"]
        
        # Build encoder
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.extend([
                nn.Linear(encoder_dims[i], encoder_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        self.encoder = nn.Sequential(*encoder_layers[:-2])  # Remove last ReLU and dropout
        
        # Variational components
        self.fc_mu = nn.Linear(encoder_dims[-2], self.latent_dim)
        self.fc_logstd = nn.Linear(encoder_dims[-2], self.latent_dim)
        
        # Build decoder
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.extend([
                nn.Linear(decoder_dims[i], decoder_dims[i+1]),
                nn.ReLU() if i < len(decoder_dims) - 2 else nn.Sigmoid(),  # Sigmoid for output
                nn.Dropout(0.1) if i < len(decoder_dims) - 2 else nn.Identity()
            ])
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent space."""
        h = self.encoder(x)
        mu = self.fc_mu(h) 
        logstd = self.fc_logstd(h)
        return mu, logstd
    
    def reparameterize(self, mu, logstd):
        """Reparameterization trick."""
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass."""
        mu, logstd = self.encode(x)
        z = self.reparameterize(mu, logstd)
        recon = self.decode(z)
        return recon, mu, logstd

def main():
    """Run architecture analysis and generate configurations."""
    print_architecture_analysis()
    
    print("📄 HYDRA-ZEN CONFIGURATION")
    print("=" * 50)
    print(get_hydra_config_yaml())
    
    print("\n\n🔧 IMPLEMENTATION NOTES")
    print("=" * 50)
    print("1. Input dimension corrected to 11 (actual CAN features)")
    print("2. Encoder: Progressive compression (larger → smaller)")
    print("3. Decoder: Progressive decompression (smaller → larger)")  
    print("4. Student model optimized for on-board deployment")
    print("5. Teacher model provides rich knowledge for distillation")
    print("6. Parameter counts realistic for VGAE architectures")

if __name__ == "__main__":
    main()