import pandas as pd
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns


column_names = ["Timestamp", "CAN_ID", "DLC", "Byte1", "Byte2", "Byte3", "Byte4", "Byte5", "Byte6", "Byte7", "Byte8", "T/R"]
df_gear = pd.read_csv("datasets/Car-Hacking Dataset/gear_dataset.csv", names=column_names).dropna()
df_dos = pd.read_csv("datasets/Car-Hacking Dataset/DoS_dataset.csv", names=column_names).dropna()
df_rpm = pd.read_csv("datasets/Car-Hacking Dataset/RPM_dataset.csv", names=column_names).dropna()
df_fuzzy = pd.read_csv("datasets/Car-Hacking Dataset/Fuzzy_dataset.csv", names=column_names).dropna()

df_gear["attack_type"] = df_gear["T/R"].apply(lambda x: 1 if x == "T" else 0)  # 1=gear
df_dos["attack_type"] = df_dos["T/R"].apply(lambda x: 2 if x == "T" else 0)  # 2 = DoS
df_rpm["attack_type"] = df_rpm["T/R"].apply(lambda x: 3 if x == "T" else 0)  # 3= RPM
df_fuzzy["attack_type"] = df_fuzzy["T/R"].apply(lambda x: 4 if x == "T" else 0)  # 4 = Fuzzy

# merge
df = pd.concat([df_gear, df_dos, df_rpm, df_fuzzy], ignore_index=True)
df.drop(columns=["T/R"], inplace=True)

df.dropna(inplace=True)
def hex_to_int(hex_str):
    try:
        return int(hex_str, 16)  
    except:
        return None  
df["CAN_ID"] = df["CAN_ID"].apply(hex_to_int)
byte_cols = ["Byte1", "Byte2", "Byte3", "Byte4", "Byte5", "Byte6", "Byte7", "Byte8"]
for col in byte_cols:
    df[col] = df[col].apply(hex_to_int)
df.dropna(inplace=True)

df_sampled = df.sample(frac=0.05, random_state=42)

# make sure  df_sampled exist
if df_sampled.empty:
    print("error smample empty")
    exit()

#stanarization
features = ["CAN_ID"] + byte_cols 
scaler = StandardScaler()
df_sampled[features] = scaler.fit_transform(df_sampled[features])

print("\npreprocessing finished df_sampled 的列：")
print(df_sampled.columns)
print(df_sampled.head())

#Dimensionality Reduction
reducer = umap.UMAP(n_neighbors=5, min_dist=0.2, metric="euclidean", low_memory=True)
embedding = reducer.fit_transform(df_sampled[features])

if embedding.shape[0] == 0:
    print("embedding empty")
    exit()


df_sampled["UMAP1"] = embedding[:, 0]
df_sampled["UMAP2"] = embedding[:, 1]

print("/nUMAP success and the column is ")
print(df_sampled.columns)
print(df_sampled.head())


plt.figure(figsize=(10, 7))
sns.scatterplot(x="UMAP1", y="UMAP2", hue=df_sampled["attack_type"].astype(str), data=df_sampled, palette="tab10", alpha=0.7)
plt.title("UMAP Projection of CAN Data (Different Attack Types)")
plt.legend(title="Attack Type")
plt.show()
