import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, Subset
import numpy as np
import os
import time

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import umap
import plotly.express as px
import pandas as pd

from preprocessing import graph_creation, GraphDataset
from models import GATWithJK  

########################################
# 1) Distillation loss (soft + hard)
########################################
def distill_loss_fn(student_logits, teacher_logits, labels, alpha=0.5, T=2.0):
    teacher_prob = torch.sigmoid(teacher_logits / T)
    student_prob = torch.sigmoid(student_logits / T)
    distill_loss = F.mse_loss(student_prob, teacher_prob)
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)
    return alpha * distill_loss + (1 - alpha) * hard_loss

def distill_train_one_epoch(teacher_model, student_model, loader, optimizer, device,
                            alpha=0.5, temperature=2.0):
    teacher_model.eval()
    student_model.train()

    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            teacher_out = teacher_model(data)
        student_out = student_model(data)

        teacher_logits = teacher_out.view(-1)
        student_logits = student_out.view(-1)
        label = data.y.float().to(device).view(-1)

        loss = distill_loss_fn(student_logits, teacher_logits, label,
                               alpha=alpha, T=temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

########################################
# 2) Student pure BCE training (no teacher)
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
# 3) Evaluate teacher or student GNN
########################################
def evaluate_gnn(model, loader, device, desc="[Model]"):
    import time
    model.eval()
    all_preds, all_labels = [], []
    start = time.time()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).view(-1)
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
# 4) UMAP: gather GNN embeddings & reduce
########################################
def gather_embeddings(model, loader, device):
    """
    用 model 对 loader 里的每个图 forward, 收集 (N,1) logits 作为 embedding
    Return:
      embeddings: shape=(N,1) numpy
      labels: shape=(N,)  (0/1)
    """
    model.eval()
    all_embs = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)  # (1,1)
            emb = out.view(-1).cpu().numpy()  # => shape=(1,)
            all_embs.append(emb)
            all_labels.append(data.y.item())
    embeddings = np.concatenate(all_embs, axis=0).reshape(-1,1)  # (N,1)
    labels = np.array(all_labels)
    return embeddings, labels

def visualize_umap_from_embeddings(embeddings, labels, title="UMAP"):
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_2d = reducer.fit_transform(embeddings)
    df_vis = pd.DataFrame(umap_2d, columns=["UMAP1","UMAP2"])
    # 不能用 map() => 需要自己转换
    df_vis["label"] = np.where(labels==0, "Normal", "Attack")
    fig = px.scatter(
        df_vis, x="UMAP1", y="UMAP2", color="label",
        title=title
    )
    fig.show()

########################################
# 5) 画原始数据 (Normal vs Attack) => UMAP
########################################
def visualize_raw_data_umap(csv_path_list):
    """
    读取多csv( normal, dos, fuzzy, rpm... ), 只做 R/T => 0/1
    采样5%，StandardScaler，UMAP 2D => plot
    """
    column_names = ["Timestamp","CAN_ID","DLC","Byte1","Byte2","Byte3","Byte4",
                    "Byte5","Byte6","Byte7","Byte8","T/R"]
    df_list = []
    for path in csv_path_list:
        df_temp = pd.read_csv(path, names=column_names, header=None)
        # 去除列数!=12的行
        df_temp = df_temp[df_temp.columns[:12]]
        df_temp.dropna(inplace=True)
        # T/R => 0/1
        df_temp["attack"] = df_temp["T/R"].map({"R":0,"T":1})
        df_list.append(df_temp)

    df_all = pd.concat(df_list, ignore_index=True)
    # hex-> int
    def hex_to_int_safe(x):
        try:
            return int(x,16)
        except:
            return None
    df_all["CAN_ID"] = df_all["CAN_ID"].apply(hex_to_int_safe)
    for b in ["Byte1","Byte2","Byte3","Byte4","Byte5","Byte6","Byte7","Byte8"]:
        df_all[b] = df_all[b].apply(hex_to_int_safe)
    df_all.dropna(inplace=True)

    # 采样5%
    df_sam = df_all.sample(frac=0.05, random_state=42)
    feats = ["CAN_ID","Byte1","Byte2","Byte3","Byte4","Byte5","Byte6","Byte7","Byte8"]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_sam[feats] = scaler.fit_transform(df_sam[feats])

    # UMAP
    red = umap.UMAP(n_components=2, random_state=42)
    emb = red.fit_transform(df_sam[feats])
    df_sam["UMAP1"] = emb[:,0]
    df_sam["UMAP2"] = emb[:,1]

    df_sam["label"] = np.where(df_sam["attack"]==0, "Normal", "Attack")
    fig = px.scatter(
        df_sam, x="UMAP1", y="UMAP2", color="label",
        title="Raw CAN Data (Normal vs Attack) ~ UMAP 2D"
    )
    fig.show()

########################################
# 6) Main
########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # A) 先画“原始数据 UMAP图”，只分 Normal/Attack
    csv_paths = [
        "datasets/Car-Hacking Dataset/Fuzzy_dataset.csv",
        "datasets/Car-Hacking Dataset/DoS_dataset.csv",
        "datasets/Car-Hacking Dataset/gear_dataset.csv",
        "datasets/Car-Hacking Dataset/RPM_dataset.csv"
    ]
    visualize_raw_data_umap(csv_paths)

    # B) 加载 GraphDataset => 训练 Student
    combined = True
    path = ""
    dataset = graph_creation(combined, path, window_size=50, stride=50)
    print("Total combined graphs in dataset:", len(dataset))

    # 只用 5% => 你想的话
    total_size = len(dataset)
    sub_size   = int(0.05 * total_size)
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

    # Teacher
    teacher = GATWithJK(10,32,1,num_layers=4,heads=8).to(device)
    teacher.load_state_dict(torch.load("D:/quadeer/task1/model/best_model(2).pth", map_location=device))
    for param in teacher.parameters():
        param.requires_grad = False
    teacher_acc, teacher_f1 = evaluate_gnn(teacher, test_loader, device, desc="[Teacher]")
    print(f"[Teacher] Test Acc={teacher_acc:.4f}, F1={teacher_f1:.4f}")

    # Student
    student = GATWithJK(10,8,1,num_layers=2,heads=2).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    # 1) Pretrain
    pretrain_epochs = 3
    for epoch in range(pretrain_epochs):
        loss = student_train_one_epoch(student, train_loader, optimizer, device)
        test_acc, test_f1 = evaluate_gnn(student, test_loader, device, desc="[Student Pretrain]")
        print(f"[Pretrain Ep {epoch+1}] Loss={loss:.4f} "
              f"| TestAcc={test_acc:.4f}, F1={test_f1:.4f}")

    # ---- 在这里可视化 "学生 Pretrain 后" 的图嵌入 ----
    emb_p, lbl_p = gather_embeddings(student, test_loader, device)
    visualize_umap_from_embeddings(emb_p, lbl_p, title="Student Embeddings after Pretrain")

    # 2) Distillation
    distill_epochs = 4
    alpha = 0.6
    temperature = 3.0
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
        print(f"[Distill Ep {epoch+1}] Loss={loss:.4f} "
              f"| TestAcc={test_acc:.4f}, F1={test_f1:.4f}")

    # ---- 在这里可视化 "学生 Distill 结束后" 的图嵌入 ----
    emb_d, lbl_d = gather_embeddings(student, test_loader, device)
    visualize_umap_from_embeddings(emb_d, lbl_d, title="Student Embeddings after Distillation")

    # Save
    # torch.save(student.state_dict(), "D:/quadeer/task1/model/distilled_small_gat.pth")
    # print("Saved student model => 'distilled_small_gat.pth'")

if __name__ == "__main__":
    main()
