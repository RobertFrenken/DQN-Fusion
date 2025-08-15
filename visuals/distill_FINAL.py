import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, f1_score
from preprocessing import graph_creation, GraphDataset
from models.models import GATWithJK


########################################
# 1) Distillation loss (soft + hard)   #
########################################
def distill_loss_fn(student_logits, teacher_logits, labels, alpha=0.5, T=2.0):
    """
    teacher_logits, student_logits: shape=(batch_size=1,), raw logits
    labels: 0 or 1
    alpha:  soft/hard label loss trade-off
    T: temperature
    """
    # soft label
    teacher_prob = torch.sigmoid(teacher_logits / T)
    student_prob = torch.sigmoid(student_logits / T)
    distill_loss = F.mse_loss(student_prob, teacher_prob)

    # hard label
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)

    # combine
    return alpha * distill_loss + (1 - alpha) * hard_loss


########################################
# 2) Distillation single epoch         #
########################################
def distill_train_one_epoch(teacher_model, student_model, loader, optimizer, device,
                            alpha=0.5, temperature=2.0):
    teacher_model.eval()
    student_model.train()

    total_loss = 0.0
    for data in loader:
        data = data.to(device)

        # teacher forward
        with torch.no_grad():
            teacher_out = teacher_model(data)  # => shape=(1,1)

        # student forward
        student_out = student_model(data)      # => shape=(1,1)

        teacher_logits = teacher_out.view(-1)
        student_logits = student_out.view(-1)
        label = data.y.float().to(device).view(-1)

        # distill loss
        loss = distill_loss_fn(student_logits, teacher_logits, label,
                               alpha=alpha, T=temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


########################################
# 3) Student pure BCE training         #
########################################
def student_train_one_epoch(student_model, loader, optimizer, device):
    student_model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)

        out = student_model(data).view(-1)
        label = data.y.float().to(device).view(-1)

        loss = F.binary_cross_entropy_with_logits(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


########################################
# 4) Evaluate (common for teacher & student GAT)
########################################
def evaluate_gnn(model, loader, device, desc="[Model]"):
    import time
    model.eval()
    all_preds, all_labels = [], []
    start = time.time()

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).view(-1)  # => shape=(1,)
            pred = (out > 0).long().cpu().numpy()
            label = data.y.cpu().numpy()
            all_preds.append(pred.item())
            all_labels.append(label.item())

    end = time.time()
    total_time = end - start
    num_samples = len(loader.dataset)
    avg_time_ms = (total_time / num_samples) * 1000

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='binary')

    print(f"{desc} Inference Time => Total: {total_time:.2f}s | Avg/sample: {avg_time_ms:.2f} ms")
    return acc, f1


########################################
# 5) Main
########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Load dataset
    combined = True
    path = ""
    dataset = graph_creation(combined, path, window_size=50, stride=50)
    print("Total combined graphs in dataset:", len(dataset))

    # 只用10%
    total_size = len(dataset)
    sub_size   = int(0.1 * total_size)
    indices    = torch.randperm(total_size)[:sub_size]
    from torch.utils.data import Subset
    subset_dataset = Subset(dataset, indices)

    train_ratio = 0.7
    train_size = int(train_ratio * sub_size)
    test_size  = sub_size - train_size
    train_dataset, test_dataset = random_split(subset_dataset, [train_size, test_size])
    print(f"Training set size: {train_size}, Test set size: {test_size}")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False)

    # 2) Teacher: big GAT
    teacher = GATWithJK(
        in_channels=10,
        hidden_channels=32,
        out_channels=1,
        num_layers=4,
        heads=8
    ).to(device)
    teacher.load_state_dict(torch.load("D:/quadeer/task1/model/best_model(2).pth", map_location=device))
    for param in teacher.parameters():
        param.requires_grad = False

    teacher_acc, teacher_f1 = evaluate_gnn(teacher, test_loader, device, desc="[Teacher]")
    print(f"[Teacher] Test Acc={teacher_acc:.4f}, F1={teacher_f1:.4f}")


    # 3) Student: small GAT
    student = GATWithJK(
        in_channels=10,
        hidden_channels=8,
        out_channels=1,
        num_layers=2,
        heads=2
    ).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)


    ############################
    # Stage 1: pretrain student
    ############################
    pretrain_epochs = 3
    print("\n=== Stage 1: Pretraining Student (BCE only, no teacher) ===")

    for epoch in range(pretrain_epochs):
        loss = student_train_one_epoch(student, train_loader, optimizer, device)
        test_acc, test_f1 = evaluate_gnn(student, test_loader, device, desc="[Student Pretrain]")
        print(f"[Pretrain Ep {epoch+1}/{pretrain_epochs}] Loss={loss:.4f} "
              f"| TestAcc={test_acc:.4f}, F1={test_f1:.4f}")


    ############################
    # Stage 2: Distillation
    ############################
    distill_epochs = 4
    alpha = 0.6
    temperature = 3.0

    print("\n=== Stage 2: Distillation (with Teacher) ===")
    for epoch in range(distill_epochs):
        loss = distill_train_one_epoch(
            teacher_model=teacher,
            student_model=student,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha=alpha,
            temperature=temperature
        )
        test_acc, test_f1 = evaluate_gnn(student, test_loader, device, desc="[Student Distill]")
        print(f"[Distill Ep {epoch+1}/{distill_epochs}] Loss={loss:.4f} "
              f"| TestAcc={test_acc:.4f}, F1={test_f1:.4f}")


    # Final
    final_acc, final_f1 = evaluate_gnn(student, test_loader, device, desc="[Student Final]")
    print(f"\nFinal [Student] Acc={final_acc:.4f}, F1={final_f1:.4f}")

    # Save
    torch.save(student.state_dict(), "D:/quadeer/task1/model/distilled_small_gat.pth")

    print("Small GAT student model saved as 'distilled_small_gat.pth'")

if __name__ == "__main__":
    main()
