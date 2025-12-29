import pandas as pd
import os
input_dir = "D:/quadeer/task1/dataset"
output_dir = os.path.join(input_dir, "graphdata")
os.makedirs(output_dir, exist_ok=True)
filenames = [
    "DoS_dataset.csv",
    "Fuzzy_dataset.csv",
    "gear_dataset.csv",
    "normal_run_data.csv",
    "RPM_dataset.csv"
]
def is_hex(s):
    try:
        int(s, 16)
        return True
    except:
        return False
def safe_hex(x):
    try:
        return int(x, 16)
    except:
        return 0

for fname in filenames:
    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, f"preprocessed_{fname}")

    cleaned_rows = []
    with open(input_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(',')

        if len(parts) == 0:
            continue


        label_idx = -1
        for i, val in enumerate(parts):
            if val.strip() in ('R', 'T'):
                label_idx = i
                break

     
        if label_idx == -1 or label_idx > 11:
            continue

  
        while label_idx < 11:
            parts.insert(label_idx, '00')
            label_idx += 1

        parts = parts[:12]

  
        if len(parts) == 12 and parts[-1] in ('R', 'T'):
            cleaned_rows.append(parts)


    columns = ['Timestamp', 'CAN_ID', 'DLC'] + [f'Byte{i}' for i in range(8)] + ['Label']
    df = pd.DataFrame(cleaned_rows, columns=columns)

    
    for col in ['CAN_ID'] + [f'Byte{i}' for i in range(8)]:
        df[col] = df[col].apply(safe_hex)

    # R → 0, T → 1
    df['Label'] = df['Label'].map({'R': 0, 'T': 1}).astype(int)

    df.to_csv(output_path, index=False)
    print(f" {fname} saved to ：{output_path}")
