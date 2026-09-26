# for data manipulation
import pandas as pd
# for creating a folder
import os
# for splitting data into train/test sets
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_REPO = "ASNaik/tourism-package-prediction"
DATASET_PATH = f"hf://datasets/{DATASET_REPO}/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print("Shape:", df.shape)

# Drop unique identifier / row-index columns that carry no predictive value
drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
df.drop(columns=drop_cols, inplace=True)

# Fix inconsistent category labels found during EDA
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

target_col = "ProdTaken"

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform a stratified train-test split (ProdTaken is imbalanced: ~19% positive class)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to the Hugging Face Hub.")
