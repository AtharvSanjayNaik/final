from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

SPACE_REPO = "ASNaik/tourism-prediction-app"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the Space already exists; create it if not
try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print(f"Space '{SPACE_REPO}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{SPACE_REPO}' not found. Creating new Space...")
    create_repo(repo_id=SPACE_REPO, repo_type="space", space_sdk="streamlit", private=False)
    print(f"Space '{SPACE_REPO}' created.")

# Step 2: Push the deployment files (app.py, requirements.txt)
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id=SPACE_REPO,                            # the target repo
    repo_type="space",                             # dataset, model, or space
    path_in_repo="",                               # optional: subfolder path inside the repo
)
