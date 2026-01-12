"""
Guided Training Sequence

This script helps you train models in the correct order with interactive guidance.
It ensures dependencies are met and provides recommendations for training sequence.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Add project root to Python path  
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train_individual_model import IndividualModelTrainer

class GuidedTrainingSequence:
    """Interactive training sequence with dependency management."""
    
    def __init__(self):
        self.trainer = IndividualModelTrainer()
        self.datasets = ['hcrl_ch', 'hcrl_sa', 'set_01', 'set_02', 'set_03', 'set_04']
        
    def get_training_status(self) -> Dict[str, Dict[str, bool]]:
        """Get current training status for all models."""
        status = {}
        
        for dataset in self.datasets:
            status[dataset] = {
                'teacher': Path(f"saved_models/best_teacher_model_{dataset}.pth").exists(),
                'student': Path(f"saved_models/final_student_model_{dataset}.pth").exists(),
                'fusion': Path(f"saved_models/fusion_agent_{dataset}.pth").exists()
            }
        
        return status
    
    def get_recommended_sequence(self) -> List[Dict[str, str]]:
        """Get recommended training sequence based on current status."""
        status = self.get_training_status()
        sequence = []
        
        # Priority order for datasets (you can modify this)
        priority_order = ['hcrl_ch', 'hcrl_sa', 'set_01', 'set_02', 'set_03', 'set_04']
        
        # First, train all missing teacher models
        for dataset in priority_order:
            if not status[dataset]['teacher']:
                sequence.append({
                    'dataset': dataset,
                    'model': 'teacher',
                    'reason': 'Base model required for knowledge distillation',
                    'estimated_minutes': 45
                })
        
        # Then, train student models where teacher exists but student doesn't
        for dataset in priority_order:
            if status[dataset]['teacher'] and not status[dataset]['student']:
                sequence.append({
                    'dataset': dataset,
                    'model': 'student', 
                    'reason': 'Knowledge distillation using trained teacher',
                    'estimated_minutes': 35
                })
        
        # Finally, train fusion models where both teacher and student exist
        for dataset in priority_order:
            if (status[dataset]['teacher'] and 
                status[dataset]['student'] and 
                not status[dataset]['fusion']):
                sequence.append({
                    'dataset': dataset,
                    'model': 'fusion',
                    'reason': 'DQN fusion using both teacher and student',
                    'estimated_minutes': 60
                })
        
        return sequence
    
    def display_status(self):
        """Display current training status in a nice format."""
        status = self.get_training_status()
        
        print("📊 Current Model Status:")
        print("=" * 60)
        print(f"{'Dataset':<12} {'Teacher':<10} {'Student':<10} {'Fusion':<10}")
        print("-" * 60)
        
        for dataset in self.datasets:
            teacher_status = "✅" if status[dataset]['teacher'] else "❌"
            student_status = "✅" if status[dataset]['student'] else "❌" 
            fusion_status = "✅" if status[dataset]['fusion'] else "❌"
            
            print(f"{dataset:<12} {teacher_status:<10} {student_status:<10} {fusion_status:<10}")
        
        # Summary
        total_models = len(self.datasets) * 3
        trained_models = sum(sum(dataset.values()) for dataset in status.values())
        completion = trained_models / total_models * 100
        
        print("-" * 60)
        print(f"Overall Progress: {trained_models}/{total_models} models ({completion:.1f}%)")
    
    def display_recommended_sequence(self):
        """Display the recommended training sequence."""
        sequence = self.get_recommended_sequence()
        
        if not sequence:
            print("\n🎉 All models are already trained!")
            return
        
        print(f"\n🎯 Recommended Training Sequence ({len(sequence)} models):")
        print("=" * 70)
        
        total_time = 0
        for i, step in enumerate(sequence, 1):
            total_time += step['estimated_minutes']
            print(f"{i:2d}. {step['dataset']:<12} {step['model']:<8} "
                  f"({step['estimated_minutes']:2d}m) - {step['reason']}")
        
        print("-" * 70)
        print(f"Total estimated time: {total_time} minutes ({total_time/60:.1f} hours)")
    
    def interactive_training(self):
        """Interactive training with user confirmation for each step."""
        sequence = self.get_recommended_sequence()
        
        if not sequence:
            print("\n🎉 All models are already trained!")
            return
        
        print(f"\n🚀 Starting Interactive Training")
        print("You can skip any step or stop at any time.")
        print("=" * 50)
        
        completed = []
        failed = []
        
        for i, step in enumerate(sequence, 1):
            print(f"\nStep {i}/{len(sequence)}: {step['dataset']} {step['model']}")
            print(f"Reason: {step['reason']}")
            print(f"Estimated time: {step['estimated_minutes']} minutes")
            
            while True:
                choice = input("\nOptions: (t)rain, (s)kip, (q)uit, (d)ry-run: ").lower().strip()
                
                if choice in ['q', 'quit']:
                    print("Training stopped by user.")
                    break
                
                elif choice in ['s', 'skip']:
                    print(f"Skipped {step['dataset']} {step['model']}")
                    break
                
                elif choice in ['d', 'dry-run']:
                    result = self.trainer.train_model(
                        dataset=step['dataset'],
                        model_type=step['model'],
                        dry_run=True
                    )
                    continue
                
                elif choice in ['t', 'train']:
                    print(f"\n🏋️ Training {step['dataset']} {step['model']}...")
                    
                    result = self.trainer.train_model(
                        dataset=step['dataset'],
                        model_type=step['model'],
                        force_retrain=False
                    )
                    
                    if result['status'] == 'completed':
                        completed.append(step)
                        print(f"✅ Successfully trained {step['dataset']} {step['model']}")
                    else:
                        failed.append((step, result))
                        print(f"❌ Failed to train {step['dataset']} {step['model']}")
                        
                        retry = input("Retry this model? (y/n): ").lower().strip()
                        if retry == 'y':
                            continue
                    
                    break
                
                else:
                    print("Invalid option. Please choose: (t)rain, (s)kip, (q)uit, (d)ry-run")
            
            if choice in ['q', 'quit']:
                break
        
        # Summary
        print(f"\n📈 Training Session Summary")
        print("=" * 40)
        print(f"Completed: {len(completed)} models")
        print(f"Failed: {len(failed)} models")
        
        if completed:
            print(f"\n✅ Successfully trained:")
            for step in completed:
                print(f"  • {step['dataset']} {step['model']}")
        
        if failed:
            print(f"\n❌ Failed to train:")
            for step, result in failed:
                error = result.get('error', 'Unknown error')
                print(f"  • {step['dataset']} {step['model']} - {error}")

def main():
    print("🎯 Guided Training Sequence for CAN-Graph Models")
    print("=" * 50)
    
    try:
        guide = GuidedTrainingSequence()
    except Exception as e:
        print(f"❌ Failed to initialize training guide: {e}")
        return 1
    
    # Show current status
    guide.display_status()
    guide.display_recommended_sequence()
    
    # Ask if user wants to proceed with interactive training
    if input(f"\nProceed with interactive training? (y/n): ").lower().strip() == 'y':
        guide.interactive_training()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())