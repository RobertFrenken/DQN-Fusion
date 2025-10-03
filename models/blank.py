import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
import pickle
import os

class QFusionAgent:
    """
    Q-learning agent for dynamic fusion of GAT and VGAE outputs.
    
    This agent learns optimal fusion weights based on the confidence levels
    of both the anomaly detection (VGAE) and classification (GAT) components.
    """
    
    def __init__(self, alpha_steps: int = 21, state_bins: int = 10, 
                 lr: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1,
                 epsilon_decay: float = 0.995, min_epsilon: float = 0.01):
        """
        Initialize Q-learning fusion agent.
        
        Args:
            alpha_steps: Number of discrete fusion weights (0.0, 0.05, ..., 1.0)
            state_bins: Number of bins for discretizing state features
            lr: Learning rate for Q-learning updates
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            min_epsilon: Minimum epsilon value
        """
        self.alpha_values = np.linspace(0, 1, alpha_steps)
        self.state_bins = state_bins
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-table: [anomaly_score_bin, gat_prob_bin, alpha_action]
        self.Q = np.zeros((state_bins, state_bins, alpha_steps))
        
        # Experience tracking
        self.experience_buffer = []
        self.reward_history = []
        self.accuracy_history = []
        
    def discretize_state(self, anomaly_score: float, gat_prob: float) -> Tuple[int, int]:
        """Discretize continuous scores into bins."""
        # Clip to [0, 1] range and discretize
        anomaly_score = np.clip(anomaly_score, 0, 1)
        gat_prob = np.clip(gat_prob, 0, 1)
        
        a_bin = min(int(anomaly_score * self.state_bins), self.state_bins - 1)
        g_bin = min(int(gat_prob * self.state_bins), self.state_bins - 1)
        
        return a_bin, g_bin
    
    def select_action(self, anomaly_score: float, gat_prob: float, 
                     training: bool = True) -> Tuple[float, int, Tuple[int, int]]:
        """
        Select fusion weight using epsilon-greedy policy.
        
        Args:
            anomaly_score: Normalized anomaly score [0, 1]
            gat_prob: GAT probability [0, 1]
            training: Whether in training mode (affects exploration)
            
        Returns:
            Tuple of (alpha_value, action_index, state_bins)
        """
        a_bin, g_bin = self.discretize_state(anomaly_score, gat_prob)
        
        if training and np.random.rand() < self.epsilon:
            # Exploration: random action
            action_idx = np.random.randint(len(self.alpha_values))
        else:
            # Exploitation: best known action
            action_idx = np.argmax(self.Q[a_bin, g_bin])
        
        alpha_value = self.alpha_values[action_idx]
        return alpha_value, action_idx, (a_bin, g_bin)
    
    def compute_reward(self, prediction: int, true_label: int, 
                      anomaly_score: float, gat_prob: float, 
                      confidence_bonus: bool = True) -> float:
        """
        Compute reward based on prediction correctness and confidence.
        
        Args:
            prediction: Model prediction (0 or 1)
            true_label: Ground truth label (0 or 1)
            anomaly_score: Anomaly detection score
            gat_prob: GAT probability
            confidence_bonus: Whether to add confidence-based bonus
            
        Returns:
            Reward value
        """
        # Base reward for correctness
        base_reward = 1.0 if prediction == true_label else -1.0
        
        if not confidence_bonus:
            return base_reward
        
        # Confidence bonus: reward high-confidence correct predictions more
        if prediction == true_label:
            # For correct predictions, reward confidence
            if prediction == 1:  # Attack correctly identified
                confidence = max(anomaly_score, gat_prob)
            else:  # Normal correctly identified  
                confidence = 1.0 - max(anomaly_score, gat_prob)
            
            confidence_reward = 0.5 * confidence
            return base_reward + confidence_reward
        else:
            # For incorrect predictions, penalize overconfidence
            if prediction == 1:  # False positive
                overconfidence = max(anomaly_score, gat_prob)
            else:  # False negative
                overconfidence = 1.0 - min(anomaly_score, gat_prob)
            
            confidence_penalty = -0.5 * overconfidence
            return base_reward + confidence_penalty
    
    def update_q_table(self, state: Tuple[int, int], action_idx: int, 
                      reward: float, next_state: Optional[Tuple[int, int]] = None):
        """Update Q-table using Q-learning rule."""
        a_bin, g_bin = state
        
        if next_state is not None:
            next_a_bin, next_g_bin = next_state
            best_next_q = np.max(self.Q[next_a_bin, next_g_bin])
        else:
            best_next_q = 0.0  # Terminal state
        
        # Q-learning update
        current_q = self.Q[a_bin, g_bin, action_idx]
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - current_q
        
        self.Q[a_bin, g_bin, action_idx] += self.lr * td_error
        
        # Track experience
        self.reward_history.append(reward)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def get_policy_summary(self) -> Dict:
        """Get summary of learned policy."""
        policy_matrix = np.zeros((self.state_bins, self.state_bins))
        
        for i in range(self.state_bins):
            for j in range(self.state_bins):
                best_action = np.argmax(self.Q[i, j])
                policy_matrix[i, j] = self.alpha_values[best_action]
        
        return {
            'policy_matrix': policy_matrix,
            'q_table': self.Q.copy(),
            'avg_recent_reward': np.mean(self.reward_history[-100:]) if self.reward_history else 0.0,
            'total_experiences': len(self.reward_history),
            'current_epsilon': self.epsilon
        }
    
    def save_agent(self, filepath: str):
        """Save agent state to file."""
        agent_state = {
            'Q': self.Q,
            'alpha_values': self.alpha_values,
            'state_bins': self.state_bins,
            'lr': self.lr,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'reward_history': self.reward_history,
            'accuracy_history': self.accuracy_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(agent_state, f)
        print(f"✓ Q-learning agent saved to {filepath}")
    
    def load_agent(self, filepath: str):
        """Load agent state from file."""
        with open(filepath, 'rb') as f:
            agent_state = pickle.load(f)
        
        self.Q = agent_state['Q']
        self.alpha_values = agent_state['alpha_values']
        self.state_bins = agent_state['state_bins']
        self.lr = agent_state['lr']
        self.gamma = agent_state['gamma']
        self.epsilon = agent_state['epsilon']
        self.reward_history = agent_state['reward_history']
        self.accuracy_history = agent_state.get('accuracy_history', [])
        
        print(f"✓ Q-learning agent loaded from {filepath}")


# Add these imports at the top
from models.adaptive_fusion import QFusionAgent
import matplotlib.pyplot as plt

# Add this method to your GATPipeline class
class GATPipeline:
    def __init__(self, num_ids: int, embedding_dim: int = 8, device: str = 'cpu'):
        # ... existing initialization code ...
        
        # Initialize Q-learning fusion agent
        self.fusion_agent = QFusionAgent(
            alpha_steps=21,     # 0.0, 0.05, 0.10, ..., 1.0
            state_bins=10,      # 10x10 state grid
            lr=0.1,             # Learning rate
            gamma=0.9,          # Discount factor
            epsilon=0.2,        # Initial exploration
            epsilon_decay=0.995,
            min_epsilon=0.01
        )
        
        self.fusion_enabled = False
        print(f"✓ GAT Pipeline with Q-Learning Fusion initialized")

    def train_adaptive_fusion(self, train_loader: DataLoader, 
                            epochs: int = 5, validation_interval: int = 100) -> Dict:
        """
        Train the Q-learning fusion agent using online learning.
        
        Args:
            train_loader: Training data for fusion learning
            epochs: Number of training epochs
            validation_interval: Interval for validation and epsilon decay
            
        Returns:
            Dictionary with training statistics
        """
        print(f"\n=== Training Adaptive Fusion Agent ({epochs} epochs) ===")
        
        if not self.fusion_enabled:
            print("Enabling fusion mode...")
            self.fusion_enabled = True
        
        self.autoencoder.eval()
        self.classifier.eval()
        
        training_stats = {
            'epoch_accuracies': [],
            'epoch_rewards': [],
            'epsilon_values': [],
            'fusion_weights_used': []
        }
        
        for epoch in range(epochs):
            epoch_correct = 0
            epoch_total = 0
            epoch_rewards = []
            epoch_alphas = []
            
            print(f"\nEpoch {epoch + 1}/{epochs} (ε={self.fusion_agent.epsilon:.3f})")
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                
                # Get model outputs
                with torch.no_grad():
                    # Autoencoder outputs for anomaly detection
                    cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
                        batch.x, batch.edge_index, batch.batch)
                    
                    # Compute anomaly scores
                    anomaly_scores = self._compute_normalized_anomaly_scores(
                        batch, cont_out, canid_logits, neighbor_logits)
                    
                    # GAT classifier outputs
                    gat_logits = self.classifier(batch)
                    gat_probs = torch.sigmoid(gat_logits.squeeze()).cpu().numpy()
                
                # Process each graph in the batch
                graphs = Batch.to_data_list(batch)
                true_labels = batch.y.cpu().numpy()
                
                for i, (graph, true_label, anomaly_score, gat_prob) in enumerate(
                    zip(graphs, true_labels, anomaly_scores, gat_probs)):
                    
                    # Select fusion weight
                    alpha, action_idx, state = self.fusion_agent.select_action(
                        anomaly_score, gat_prob, training=True)
                    
                    # Make fused prediction
                    fused_score = (1 - alpha) * anomaly_score + alpha * gat_prob
                    prediction = 1 if fused_score > 0.5 else 0
                    
                    # Compute reward
                    reward = self.fusion_agent.compute_reward(
                        prediction, int(true_label), anomaly_score, gat_prob)
                    
                    # Update Q-table
                    self.fusion_agent.update_q_table(state, action_idx, reward)
                    
                    # Track statistics
                    epoch_correct += (prediction == int(true_label))
                    epoch_total += 1
                    epoch_rewards.append(reward)
                    epoch_alphas.append(alpha)
                
                # Periodic validation and epsilon decay
                if batch_idx % validation_interval == 0 and batch_idx > 0:
                    self.fusion_agent.decay_epsilon()
                    
                    if batch_idx % (validation_interval * 2) == 0:
                        current_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
                        avg_reward = np.mean(epoch_rewards[-validation_interval:])
                        print(f"  Batch {batch_idx}: Acc={current_acc:.3f}, "
                              f"Reward={avg_reward:.3f}, α_avg={np.mean(epoch_alphas[-validation_interval:]):.3f}")
            
            # Epoch statistics
            epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
            epoch_avg_reward = np.mean(epoch_rewards)
            epoch_avg_alpha = np.mean(epoch_alphas)
            
            training_stats['epoch_accuracies'].append(epoch_accuracy)
            training_stats['epoch_rewards'].append(epoch_avg_reward)
            training_stats['epsilon_values'].append(self.fusion_agent.epsilon)
            training_stats['fusion_weights_used'].append(epoch_avg_alpha)
            
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Accuracy: {epoch_accuracy:.4f}")
            print(f"  Avg Reward: {epoch_avg_reward:.4f}")
            print(f"  Avg Fusion Weight: {epoch_avg_alpha:.4f}")
            print(f"  Epsilon: {self.fusion_agent.epsilon:.4f}")
        
        print(f"\n✓ Adaptive fusion training complete!")
        self._plot_fusion_training_stats(training_stats)
        
        return training_stats

    def _compute_normalized_anomaly_scores(self, batch, cont_out, canid_logits, neighbor_logits):
        """Compute normalized anomaly scores for each graph in batch."""
        # Component errors
        node_errors = (cont_out - batch.x[:, 1:]).pow(2).mean(dim=1)
        
        neighbor_targets = self.autoencoder.create_neighborhood_targets(
            batch.x, batch.edge_index, batch.batch)
        neighbor_errors = nn.BCEWithLogitsLoss(reduction='none')(
            neighbor_logits, neighbor_targets).mean(dim=1)
        
        canid_pred = canid_logits.argmax(dim=1)
        
        anomaly_scores = []
        graphs = Batch.to_data_list(batch)
        start = 0
        
        for graph in graphs:
            num_nodes = graph.x.size(0)
            
            # Graph-level errors
            node_error = node_errors[start:start+num_nodes].max().item()
            neighbor_error = neighbor_errors[start:start+num_nodes].max().item()
            
            true_canids = graph.x[:, 0].long().cpu()
            pred_canids = canid_pred[start:start+num_nodes].cpu()
            canid_error = 1.0 - (pred_canids == true_canids).float().mean().item()
            
            # Composite score
            raw_score = (
                FUSION_WEIGHTS['node_reconstruction'] * node_error +
                FUSION_WEIGHTS['neighborhood_prediction'] * neighbor_error +
                FUSION_WEIGHTS['can_id_prediction'] * canid_error
            )
            
            # Normalize to [0, 1]
            normalized_score = torch.sigmoid(torch.tensor(raw_score * 5 - 2.5)).item()
            anomaly_scores.append(normalized_score)
            
            start += num_nodes
        
        return anomaly_scores

    def predict_with_adaptive_fusion(self, data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prediction using learned adaptive fusion weights.
        
        Returns:
            Tuple of (predictions, anomaly_scores, gat_probs, fusion_weights)
        """
        data = data.to(self.device)
        
        with torch.no_grad():
            # Get model outputs
            cont_out, canid_logits, neighbor_logits, _, _ = self.autoencoder(
                data.x, data.edge_index, data.batch)
            gat_logits = self.classifier(data)
            gat_probs = torch.sigmoid(gat_logits.squeeze())
            
            # Compute anomaly scores
            anomaly_scores = self._compute_normalized_anomaly_scores(
                data, cont_out, canid_logits, neighbor_logits)
            
            predictions = []
            fusion_weights = []
            
            graphs = Batch.to_data_list(data)
            
            for i, (graph, anomaly_score, gat_prob) in enumerate(
                zip(graphs, anomaly_scores, gat_probs.cpu().numpy())):
                
                # Get learned fusion weight (no exploration during inference)
                alpha, _, _ = self.fusion_agent.select_action(
                    anomaly_score, gat_prob, training=False)
                
                # Fused prediction
                fused_score = (1 - alpha) * anomaly_score + alpha * gat_prob
                prediction = 1 if fused_score > 0.5 else 0
                
                predictions.append(prediction)
                fusion_weights.append(alpha)
        
        return (
            torch.tensor(predictions, device=self.device),
            torch.tensor(anomaly_scores),
            gat_probs.cpu(),
            torch.tensor(fusion_weights)
        )

    def evaluate_adaptive_fusion(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model using adaptive fusion and compare with fixed fusion.
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        print(f"\n=== Adaptive Fusion Evaluation ===")
        
        # Results storage
        all_labels = []
        all_adaptive_preds = []
        all_fixed_preds = []
        all_anomaly_scores = []
        all_gat_probs = []
        all_fusion_weights = []
        
        for batch in test_loader:
            batch = batch.to(self.device)
            labels = batch.y.cpu().numpy()
            
            # Adaptive fusion predictions
            adaptive_preds, anomaly_scores, gat_probs, fusion_weights = \
                self.predict_with_adaptive_fusion(batch)
            
            # Fixed fusion predictions (α=0.85)
            fixed_preds, _, _ = self.predict_with_fusion(batch, alpha=0.85)
            
            # Store results
            all_labels.extend(labels)
            all_adaptive_preds.extend(adaptive_preds.cpu().numpy())
            all_fixed_preds.extend(fixed_preds.cpu().numpy())
            all_anomaly_scores.extend(anomaly_scores.cpu().numpy())
            all_gat_probs.extend(gat_probs.cpu().numpy())
            all_fusion_weights.extend(fusion_weights.cpu().numpy())
        
        # Calculate metrics
        adaptive_accuracy = np.mean(np.array(all_adaptive_preds) == np.array(all_labels))
        fixed_accuracy = np.mean(np.array(all_fixed_preds) == np.array(all_labels))
        
        results = {
            'adaptive': {
                'accuracy': adaptive_accuracy,
                'confusion_matrix': confusion_matrix(all_labels, all_adaptive_preds),
                'report': classification_report(all_labels, all_adaptive_preds, output_dict=True)
            },
            'fixed': {
                'accuracy': fixed_accuracy,
                'confusion_matrix': confusion_matrix(all_labels, all_fixed_preds),
                'report': classification_report(all_labels, all_fixed_preds, output_dict=True)
            },
            'fusion_analysis': {
                'anomaly_scores': all_anomaly_scores,
                'gat_probs': all_gat_probs,
                'fusion_weights': all_fusion_weights,
                'labels': all_labels
            }
        }
        
        # Print comparison
        print(f"Adaptive Fusion Accuracy: {adaptive_accuracy:.4f}")
        print(f"Fixed Fusion Accuracy:    {fixed_accuracy:.4f}")
        print(f"Improvement: {(adaptive_accuracy - fixed_accuracy)*100:.2f}%")
        
        # Analyze fusion weight distribution
        avg_weight = np.mean(all_fusion_weights)
        print(f"Average Fusion Weight: {avg_weight:.3f} (vs. fixed 0.85)")
        
        # Generate analysis plots
        self._plot_fusion_analysis(results['fusion_analysis'])
        
        return results

    def _plot_fusion_training_stats(self, stats: Dict):
        """Plot training statistics for fusion learning."""
        epochs = range(1, len(stats['epoch_accuracies']) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Accuracy over epochs
        ax1.plot(epochs, stats['epoch_accuracies'], 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Fusion Learning Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # Average reward over epochs
        ax2.plot(epochs, stats['epoch_rewards'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Q-Learning Rewards')
        ax2.grid(True, alpha=0.3)
        
        # Epsilon decay
        ax3.plot(epochs, stats['epsilon_values'], 'r-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Epsilon (Exploration)')
        ax3.set_title('Exploration Rate Decay')
        ax3.grid(True, alpha=0.3)
        
        # Average fusion weights used
        ax4.plot(epochs, stats['fusion_weights_used'], 'm-', linewidth=2)
        ax4.axhline(y=0.85, color='k', linestyle='--', alpha=0.7, label='Fixed Weight')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Average Fusion Weight')
        ax4.set_title('Learned Fusion Weights')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/fusion_training_stats.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_fusion_analysis(self, fusion_data: Dict):
        """Plot detailed fusion analysis."""
        anomaly_scores = np.array(fusion_data['anomaly_scores'])
        gat_probs = np.array(fusion_data['gat_probs'])
        fusion_weights = np.array(fusion_data['fusion_weights'])
        labels = np.array(fusion_data['labels'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Fusion weight distribution
        ax1.hist(fusion_weights, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(x=np.mean(fusion_weights), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(fusion_weights):.3f}')
        ax1.axvline(x=0.85, color='green', linestyle='--', label='Fixed: 0.85')
        ax1.set_xlabel('Fusion Weight (α)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Learned Fusion Weight Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2D state space with fusion weights
        scatter = ax2.scatter(anomaly_scores, gat_probs, c=fusion_weights, 
                            cmap='viridis', alpha=0.6, s=20)
        ax2.set_xlabel('Anomaly Score')
        ax2.set_ylabel('GAT Probability')
        ax2.set_title('Learned Policy: Fusion Weights by State')
        plt.colorbar(scatter, ax=ax2, label='Fusion Weight')
        
        # Fusion weights by class
        normal_weights = fusion_weights[labels == 0]
        attack_weights = fusion_weights[labels == 1]
        
        ax3.hist([normal_weights, attack_weights], bins=15, alpha=0.7, 
                label=['Normal', 'Attack'], color=['blue', 'red'])
        ax3.set_xlabel('Fusion Weight (α)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Fusion Weights by True Class')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Policy heatmap
        policy_summary = self.fusion_agent.get_policy_summary()
        im = ax4.imshow(policy_summary['policy_matrix'], cmap='viridis', 
                       aspect='auto', origin='lower')
        ax4.set_xlabel('GAT Probability Bin')
        ax4.set_ylabel('Anomaly Score Bin')
        ax4.set_title('Learned Policy Heatmap')
        plt.colorbar(im, ax=ax4, label='Fusion Weight')
        
        plt.tight_layout()
        plt.savefig('images/fusion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_fusion_agent(self, save_folder: str, dataset_key: str):
        """Save the trained fusion agent."""
        agent_path = os.path.join(save_folder, f'fusion_agent_{dataset_key}.pkl')
        self.fusion_agent.save_agent(agent_path)
    
    def load_fusion_agent(self, save_folder: str, dataset_key: str):
        """Load a trained fusion agent."""
        agent_path = os.path.join(save_folder, f'fusion_agent_{dataset_key}.pkl')
        self.fusion_agent.load_agent(agent_path)
        self.fusion_enabled = True

def main_with_adaptive_fusion(config: DictConfig):
    """Main training pipeline with adaptive fusion learning."""
    
    # ... existing setup code ...
    
    # Initialize pipeline
    pipeline = GATPipeline(num_ids=len(id_mapping), embedding_dim=8, device=str(device))
    
    # Stage 1: Train autoencoder (existing)
    pipeline.train_stage1(train_loader, epochs=config.autoencoder_epochs)
    
    # Stage 2: Train classifier (existing)  
    pipeline.train_stage2(full_train_loader, epochs=config.classifier_epochs)
    
    # Stage 3: Train adaptive fusion
    print(f"\n=== Stage 3: Adaptive Fusion Learning ===")
    fusion_stats = pipeline.train_adaptive_fusion(
        train_loader=fusion_train_loader,  # Can reuse train_loader
        epochs=config.fusion_epochs,       # e.g., 5-10 epochs
        validation_interval=100
    )
    
    # Evaluation with comparison
    print(f"\n=== Final Evaluation with Adaptive Fusion ===")
    
    # Standard evaluation
    standard_results = pipeline.evaluate(test_loader, method='standard')
    
    # Fixed fusion evaluation
    fixed_results = pipeline.evaluate(test_loader, method='fusion')
    
    # Adaptive fusion evaluation
    adaptive_results = pipeline.evaluate_adaptive_fusion(test_loader)
    
    # Print comparison
    print(f"\n📊 FINAL RESULTS COMPARISON")
    print(f"{'='*50}")
    print(f"Standard Two-Stage:   {standard_results['standard']['accuracy']:.4f}")
    print(f"Fixed Fusion (α=0.85): {fixed_results['fusion']['accuracy']:.4f}")
    print(f"Adaptive Fusion:      {adaptive_results['adaptive']['accuracy']:.4f}")
    
    # Save models including fusion agent
    pipeline.save_models("saved_models", dataset_key, epochs, embedding_dim, len(id_mapping))
    pipeline.save_fusion_agent("saved_models", dataset_key)
    
    return {
        'standard': standard_results,
        'fixed_fusion': fixed_results,
        'adaptive_fusion': adaptive_results,
        'fusion_training': fusion_stats
    }


def test_adaptive_fusion():
    """Simple test for adaptive fusion functionality."""
    
    # Initialize fusion agent
    agent = QFusionAgent(alpha_steps=11, state_bins=5)
    
    # Simulate some interactions
    for i in range(100):
        # Random state
        anomaly_score = np.random.random()
        gat_prob = np.random.random()
        
        # Select action
        alpha, action_idx, state = agent.select_action(anomaly_score, gat_prob)
        
        # Simulate reward (higher for balanced fusion)
        optimal_alpha = 0.5 + 0.3 * (anomaly_score - gat_prob)
        reward = 1.0 - abs(alpha - optimal_alpha)
        
        # Update
        agent.update_q_table(state, action_idx, reward)
        
        if i % 20 == 0:
            agent.decay_epsilon()
    
    # Check if learning occurred
    summary = agent.get_policy_summary()
    print(f"Final average reward: {summary['avg_recent_reward']:.3f}")
    print(f"Policy learned: {summary['policy_matrix'].mean():.3f}")
    
if __name__ == "__main__":
    test_adaptive_fusion()


'''
6. Benefits of This Approach
Adaptive Learning: The fusion weights adapt to different scenarios
State-Dependent: Different fusion strategies for different confidence combinations
Continuous Improvement: The agent keeps learning from new data
Interpretable: You can visualize the learned policy
Comparative: Easy to compare against fixed fusion strategies
Start by running the test, then integrate step by step into your existing pipeline. 
The Q-learning agent will learn when to trust the GAT classifier more vs. when 
to rely more on anomaly detection based on their respective confidence levels.

'''