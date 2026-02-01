from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

def slugify(name: str) -> str:
    """Convert name to HF-compatible slug (underscores to hyphens)."""
    return name.replace("_", "-")

# Initialize API with token
api = HfApi(token=os.getenv("HF_TOKEN"))
hf_username = os.getenv("HF_USERNAME")
hf_space_name = slugify(os.getenv("HF_SPACE_NAME", "predictive-maintenance-app"))

repo_id = f"{hf_username}/{hf_space_name}"
repo_type = "space"

# Check if the space exists, create if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, space_sdk="docker", private=False)
    print(f"Space '{repo_id}' created.")

# Upload deployment folder to Hugging Face Space
api.upload_folder(
    folder_path="pm/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",
)

print(f"Deployment files uploaded to {repo_id}")
print(f"Space URL: https://huggingface.co/spaces/{repo_id}")
