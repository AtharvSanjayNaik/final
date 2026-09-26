from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "ASNaik/tourism-package-prediction"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the dataset repo already exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Step 2: Upload the raw tourism.csv (and anything else) present in the data folder
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
