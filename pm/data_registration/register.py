import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

def slugify(name: str) -> str:
    """Convert name to HF-compatible slug (underscores to hyphens)."""
    return name.replace("_", "-")

# Paths
DATA_DIR = "pm/data"

# Load configuration from environment
hf_username = os.getenv("HF_USERNAME")
hf_dataset_name = slugify(os.getenv("HF_DATASET_NAME", "predictive-maintenance-data"))
repo_id = f"{hf_username}/{hf_dataset_name}"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if the dataset repository exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

# Upload raw data file
api.upload_file(
    path_or_fileobj=f"{DATA_DIR}/engine_data.csv",
    path_in_repo="engine_data.csv",
    repo_id=repo_id,
    repo_type=repo_type,
)

print(f"Uploaded engine_data.csv to {repo_id}")
print(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")
