import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from preprocessing import graph_creation, GraphDataset
from models.models import GATWithJK
import hydra
from sklearn.utils.class_weight import compute_class_weight

#########################################
# 0) Distillation Loss Function         #
#########################################
def distillation_loss_fn(student_logits, teacher_logits, T=5.0):
    """
    Compute the distillation loss with temperature scaling.
    """
    # Clamp logits to prevent numerical instability
    teacher_logits = torch.clamp(teacher_logits, -10, 10)
    student_logits = torch.clamp(student_logits, -10, 10)

    # Compute softmax probabilities with temperature scaling
    teacher_prob = F.softmax(teacher_logits / T, dim=-1)
    student_log_prob = F.log_softmax(student_logits / T, dim=-1)

    # KL divergence loss
    distill_loss = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (T * T)
    return distill_loss
#########################################
# 0) Focal Loss Function         #
#########################################
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.0, reduction='mean'):
        """
        Focal Loss for addressing class imbalance.
        Args:
            alpha (float): Weighting factor for the minority class.
            gamma (float): Focusing parameter to reduce the loss for well-classified examples.
            reduction (str): Specifies the reduction to apply to the output ('mean', 'sum', or 'none').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute binary cross-entropy loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # Compute the probability of the correct class
        pt = torch.exp(-BCE_loss)
        # Apply the focal loss formula
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
        


def compute_class_weights(labels):
    """
    Compute class weights based on the distribution of labels.
    Args:
        labels (numpy array): Array of binary labels (0 and 1).
    Returns:
        torch.Tensor: Class weights for the minority and majority classes.
    """
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)
#########################################
# 1) PyTorch Trainer Class              #
#########################################
class PyTorchTrainer:
    def __init__(self, model, optimizer, loss_fn, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.use_focal_loss = False
        
        # Training state tracking
        self.best_val_loss = float('inf')
        self.best_model_weights = None
        self.metrics = {
            'train': {'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []},
            'val': {'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
        }
    
    def train_one_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        all_preds, all_targets = [], []

        # Use Focal Loss or BCEWithLogitsLoss based on the toggle
        if self.use_focal_loss:
            loss_fn = FocalLoss(alpha=1.0, gamma=1.0)
        else:
            loss_fn = self.loss_fn

        for batch in train_loader:
            batch = batch.to(self.device)
            targets = batch.y.float()

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch).squeeze()
            loss = loss_fn(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics collection
            epoch_loss += loss.item() * batch.size(0)
            preds = (outputs > 0).long()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        

        # Compute metrics
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        accuracy = accuracy_score(all_targets, all_preds)
        # Print metrics in a table-like format
        print(f"{'Phase':<10} | {'Loss':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
        print("-" * 60)
        print(f"{'Train':<10} | {epoch_loss / len(train_loader.dataset):<10.4f} | {accuracy:<10.4f} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f}")
        # Update training metrics
        self._update_metrics('train', epoch_loss/len(train_loader.dataset), 
                            all_preds, all_targets)

    def validate(self, val_loader):
        self.model.eval()
        epoch_loss = 0
        all_preds, all_targets = [], []

        

        # Use Focal Loss or BCEWithLogitsLoss based on the toggle
        if self.use_focal_loss:
            loss_fn = FocalLoss(alpha=1.0, gamma=1.0)
        else:
            loss_fn = self.loss_fn

        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                targets = batch.y.float()
                
                outputs = self.model(batch).squeeze()
                # loss = self.loss_fn(outputs, targets)
                loss = loss_fn(outputs, targets)
                
                epoch_loss += loss.item() * batch.size(0)
                preds = (outputs > 0).long()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Update validation metrics
        avg_loss = epoch_loss / len(val_loader.dataset)
        self._update_metrics('val', avg_loss, all_preds, all_targets)

         # Compute metrics
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        accuracy = accuracy_score(all_targets, all_preds)

        # Print metrics
        print(f"{'Validation':<10} | {epoch_loss / len(val_loader.dataset):<10.4f} | {accuracy:<10.4f} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f}")

        
        # Check for best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_model_weights = self.model.state_dict().copy()

    def compute_metrics(self, phase):
        return self.metrics[phase]

    def _update_metrics(self, phase, loss, preds, targets):
        self.metrics[phase]['loss'].append(loss)
        self.metrics[phase]['accuracy'].append(accuracy_score(targets, preds))
        self.metrics[phase]['f1'].append(f1_score(targets, preds, average='macro'))
        self.metrics[phase]['precision'].append(
            precision_score(targets, preds, average='macro', zero_division=0)
        )
        self.metrics[phase]['recall'].append(
            recall_score(targets, preds, average='macro', zero_division=0)
        )

    def report_latest_metrics(self):
        return {
        'train': {k: (v[-1] if v else None) for k, v in self.metrics['train'].items()},
        'val': {k: (v[-1] if v else None) for k, v in self.metrics['val'].items()}
        }

class PyTorchDistillationTrainer(PyTorchTrainer):
    def __init__(self, model, optimizer, loss_fn, device="cuda",
                 teacher_model=None, warmup_epochs=5, alpha=0.1,
                 distillation_loss_fn=torch.nn.KLDivLoss(reduction='batchmean'),
                 use_focal_loss=False):
        super().__init__(model, optimizer, loss_fn, device)
        
        # Distillation-specific parameters
        self.teacher_model = teacher_model
        if self.teacher_model:
            self.teacher_model = teacher_model.to(device).eval()
            
        self.warmup_epochs = warmup_epochs
        self.alpha = alpha
        self.distillation_loss_fn = distillation_loss_fn
        self.current_epoch = 0
        self.use_focal_loss = use_focal_loss
        
        # Add distillation metrics
        self.metrics['train']['student_loss'] = []
        self.metrics['train']['distill_loss'] = []

    def train_one_epoch(self, train_loader):
        self.model.train()
        epoch_student_loss = 0
        epoch_distill_loss = 0
        all_preds, all_targets = [], []

        # Use Focal Loss for the student model
        if self.use_focal_loss:
            focal_loss_fn = FocalLoss(alpha=1.0, gamma=1.0)
        else:
            focal_loss_fn = self.loss_fn

        if self.current_epoch == self.warmup_epochs:
                    print(f"Transitioning to knowledge distillation at epoch {self.current_epoch}...")
        
        for batch in train_loader:
            batch = batch.to(self.device)
            targets = batch.y.float()
            self.optimizer.zero_grad()
            
            # Forward pass
            student_outputs = self.model(batch).squeeze()
            
            if self.teacher_model and self.current_epoch >= self.warmup_epochs:
                # Distillation phase
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(batch)
                
                teacher_logits = teacher_outputs.view(-1)
                student_logits = student_outputs.view(-1)
                # Calculate combined loss
                # student_loss = self.loss_fn(student_outputs, targets)
                student_loss = focal_loss_fn(student_outputs, targets)
                # Distillation loss
                distill_loss = distillation_loss_fn(student_logits, teacher_logits, T=5.0)
                loss = (1 - self.alpha) * student_loss + self.alpha * distill_loss
                
                # Track both losses
                epoch_student_loss += student_loss.item() * batch.size(0)
                epoch_distill_loss += distill_loss.item() * batch.size(0)
            else:
                # Warmup phase or normal training
                loss = focal_loss_fn(student_outputs, targets)
                # loss = self.loss_fn(student_outputs, targets)
                epoch_student_loss += loss.item() * batch.size(0)
                epoch_distill_loss = 0  # Not applicable
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track predictions
            preds = (student_outputs > 0).long()
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
        # Compute metrics
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        accuracy = accuracy_score(all_targets, all_preds)

        # Print metrics
        print(f"Train Metrics - Loss: {(epoch_student_loss + epoch_distill_loss) / len(train_loader.dataset):.4f}, "
              f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Update metrics
        total_samples = len(train_loader.dataset)
        self.metrics['train']['student_loss'].append(epoch_student_loss / total_samples)
        self.metrics['train']['distill_loss'].append(epoch_distill_loss / total_samples) 
        
        self._update_metrics('train', 
                            (epoch_student_loss + epoch_distill_loss) / total_samples,
                            all_preds, all_targets)
        # print(f"Metrics after epoch {self.current_epoch}: {self.metrics}")
        self.current_epoch += 1

    def validate(self, val_loader):
        # Validation remains the same - only student model is evaluated
        return super().validate(val_loader)

    def report_latest_metrics(self):
        metrics = super().report_latest_metrics()
        if self.teacher_model and self.current_epoch >= self.warmup_epochs:
            if self.metrics['train']['student_loss']:
                metrics['train']['student_loss'] = self.metrics['train']['student_loss'][-1]
            if self.metrics['train']['distill_loss']:
                metrics['train']['distill_loss'] = self.metrics['train']['distill_loss'][-1]
        return metrics

def safe_fmt(val):
    return f"{val:.4f}" if val is not None else "   N/A   "
class DistillationTrainer:
    def __init__(self, student, teacher=None, device="cuda", **kwargs):
        self.student = student.to(device)
        self.teacher = teacher.to(device) if teacher else None
        self.device = device
        
        # Distillation-specific parameters
        self.distill_alpha = kwargs.get('distill_alpha', 0.5)
        self.warmup_epochs = kwargs.get('warmup_epochs', 5)
        self.teacher_epochs = kwargs.get('teacher_epochs', 10)
        self.student_epochs = kwargs.get('student_epochs', 20)
        self.lr = kwargs.get('lr', 0.0001)
        self.use_focal_loss = kwargs.get('use_focal_loss', False)  # Toggle for focal loss
        # Initialize metrics for teacher and student
        self.teacher_metrics = {"loss": [], "accuracy": []}  # Add other metrics as needed
        self.student_metrics = {"loss": [], "accuracy": []}  # Add other metrics as needed

    def train_teacher(self, train_loader, test_loader=None, max_batches=None):
            """Train the teacher model using PyTorchTrainer."""
            if not self.teacher:
                raise ValueError("Teacher model is not defined.")
            
            if self.use_focal_loss:
                loss_fn = FocalLoss(alpha=1.0, gamma=1.0)
                print("Using Focal Loss for teacher model training.")
            else:
                loss_fn = torch.nn.BCEWithLogitsLoss()
                print("Using BCE Loss for teacher model training.")
                
            teacher_trainer = PyTorchTrainer(
                model=self.teacher,
                optimizer=torch.optim.Adam(self.teacher.parameters(), lr=self.lr),
                loss_fn=loss_fn,
                device=self.device
            )

            self.best_teacher_model = None  # Initialize best_teacher_model
            best_val_loss = float('inf')  # Track the best validation loss
            
            for epoch in range(self.teacher_epochs):
                if epoch == 0:  # Print the header only once
                    print(f"{'Epoch':<10} | {'Train Loss':<10} | {'Train Acc':<10} | {'Test Loss':<10} | {'Test Acc':<10}")
                    print("-" * 60)
                if max_batches is not None:
                    batch_count = 0
                    for batch in train_loader:
                        # Simulate one batch step (forward, backward, optimizer)
                        teacher_trainer.model.train()
                        batch = batch.to(self.device)
                        targets = batch.y.float()
                        teacher_trainer.optimizer.zero_grad()
                        outputs = teacher_trainer.model(batch).squeeze()
                        loss = loss_fn(outputs, targets)
                        loss.backward()
                        teacher_trainer.optimizer.step()
                        batch_count += 1
                        if batch_count >= max_batches:
                            break
                    teacher_trainer.validate(test_loader)
                else:
                    teacher_trainer.train_one_epoch(train_loader)
                    teacher_trainer.validate(test_loader)
                metrics_train = teacher_trainer.report_latest_metrics()['train']
                metrics_test = teacher_trainer.report_latest_metrics()['val']
                print(f"Teacher Epoch {epoch+1} | Train Loss: {safe_fmt(metrics_train['loss'])} | "
                  f"Train Acc: {safe_fmt(metrics_train['accuracy'])} | "
                  f"Test Loss: {safe_fmt(metrics_test['loss'])} | "
                  f"Test Acc: {safe_fmt(metrics_test['accuracy'])}")
                
                # Save the best teacher model
                if teacher_trainer.best_val_loss < best_val_loss:
                    best_val_loss = teacher_trainer.best_val_loss
                    self.best_teacher_model = teacher_trainer.model.state_dict().copy()
            
            # save metrics, which will be dumped to json
            self.teacher_metrics = {
            "loss": teacher_trainer.metrics['train']['loss'],
            "accuracy": teacher_trainer.metrics['train']['accuracy']
            }
    
    def train_student(self, train_loader, test_loader=None, max_batches=None):
        """Train the student model using PyTorchDistillationTrainer."""
        student_trainer = PyTorchDistillationTrainer(
            model=self.student,
            optimizer=torch.optim.Adam(self.student.parameters(), lr=self.lr),
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            teacher_model=self.teacher,
            warmup_epochs=self.warmup_epochs,
            alpha=self.distill_alpha,
            device=self.device,
            use_focal_loss=self.use_focal_loss  # Pass the toggle
        )
        
        for epoch in range(self.student_epochs):
            if max_batches is not None:
                batch_count = 0
                for batch in train_loader:
                    student_trainer.model.train()
                    batch = batch.to(self.device)
                    targets = batch.y.float()
                    student_trainer.optimizer.zero_grad()
                    outputs = student_trainer.model(batch).squeeze()
                    if student_trainer.use_focal_loss:
                        loss_fn = FocalLoss(alpha=1.0, gamma=1.0)
                    else:
                        loss_fn = student_trainer.loss_fn
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    student_trainer.optimizer.step()
                    batch_count += 1
                    if batch_count >= max_batches:
                        break
                student_trainer.validate(test_loader)
            else:
                student_trainer.train_one_epoch(train_loader)
                student_trainer.validate(test_loader)
            metrics_train = student_trainer.report_latest_metrics()['train']
            metrics_test = student_trainer.report_latest_metrics()['val']
            print(f"Student Epoch {epoch+1} | Train Loss: {safe_fmt(metrics_train['loss'])} | "
                  f"Train Acc: {safe_fmt(metrics_train['accuracy'])} | "
                  f"Test Loss: {safe_fmt(metrics_test['loss'])} | "
                  f"Test Acc: {safe_fmt(metrics_test['accuracy'])}")
            
        # save metrics, which will be dumped to json
        self.student_metrics = {
        "loss": student_trainer.metrics['train']['loss'],
        "accuracy": student_trainer.metrics['train']['accuracy']
        }


    def train_sequential(self, train_loader, test_loader=None, max_batches=None):
        """Train the teacher first, then the student."""
        if self.teacher:
            print("Training teacher model...")
            self.train_teacher(train_loader, test_loader, max_batches=max_batches)
        
        print("Training student model...")
        self.train_student(train_loader, test_loader, max_batches=max_batches)


