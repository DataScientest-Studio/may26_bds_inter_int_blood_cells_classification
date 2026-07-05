import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# The main dataset directory
dataset_dir = r"G:\Liora\PBC_dataset_normal_DIB"

# Explicit singular lowercase names matching the hard drive folders
classes = ['basophil', 
           'eosinophil', 
           'erythroblast', 
           'ig', 
           'lymphocyte', 
           'monocyte', 
           'neutrophil', 
           'platelet'
           ]

data = {}
for cls in classes:
    full_path = os.path.join(dataset_dir, cls)
    if os.path.exists(full_path):
        data[cls] = len(os.listdir(full_path))
    else:
        data[cls] = "Path missing! Double-check spelling."

# Build and display table
df = pd.DataFrame(list(data.items()), columns=['Cell_Type', 'Count'])
print("\n=== Dataset Distribution ===")
print(df)

df = df.sort_values("Count", ascending=False)

print(df)

plt.figure(figsize=(10,5))
sns.barplot(data=df, x="Cell_Type", y="Count")

plt.title("Blood Cell Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()