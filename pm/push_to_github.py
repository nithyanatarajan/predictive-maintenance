"""
Push pm folder to GitHub repository.
Creates repo if needed, copies workflow file, and pushes all files.
"""
import os
import subprocess
import shutil
import tempfile
from pathlib import Path

def slugify(name: str) -> str:
    """Convert name to HF-compatible slug."""
    return name.replace("_", "-")

# Load configuration from environment
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
GITHUB_REPO = os.getenv("GITHUB_REPO", "predictive-maintenance-pipeline")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not all([GITHUB_USERNAME, GITHUB_REPO, GITHUB_TOKEN]):
    raise ValueError("GITHUB_USERNAME, GITHUB_REPO, and GITHUB_TOKEN must be set")

# Paths
PROJECT_DIR = Path("pm")
WORKFLOW_SRC = PROJECT_DIR / ".github" / "workflows" / "pipeline.yml"
WORKFLOW_DST = Path(".github") / "workflows" / "pipeline.yml"

def run_cmd(cmd, cwd=None, safe_cmd=None):
    """Run shell command and return output."""
    display_cmd = safe_cmd if safe_cmd else cmd
    print(f"Running: {display_cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"Command failed: {display_cmd}")
    return result.stdout.strip()

# Create temporary directory for git operations
with tempfile.TemporaryDirectory() as tmpdir:
    repo_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"

    # Try to clone existing repo, or initialize new one
    try:
        run_cmd(f"git clone {repo_url} {tmpdir}", safe_cmd=f"git clone https://***@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git")
        print(f"Cloned existing repo: {GITHUB_REPO}")
    except RuntimeError:
        print(f"Creating new repo: {GITHUB_REPO}")
        # Create repo via GitHub API
        run_cmd(f'curl -X POST -H "Authorization: token {GITHUB_TOKEN}" '
                f'-H "Accept: application/vnd.github.v3+json" '
                f'https://api.github.com/user/repos -d \'{{"name":"{GITHUB_REPO}","private":false}}\'')

        # Initialize local repo
        run_cmd(f"git init {tmpdir}")
        run_cmd(f"git remote add origin {repo_url}", cwd=tmpdir, safe_cmd=f"git remote add origin https://***@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git")

    # Copy pm folder contents
    for item in PROJECT_DIR.iterdir():
        if item.name == ".github":
            continue  # Handle separately
        dst = Path(tmpdir) / "pm" / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)

    # Copy workflow file to root .github/workflows/
    workflow_dst = Path(tmpdir) / ".github" / "workflows"
    workflow_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WORKFLOW_SRC, workflow_dst / "pipeline.yml")

    # Copy data folder
    data_src = Path("pm/data")
    if data_src.exists():
        shutil.copytree(data_src, Path(tmpdir) / "pm" / "data", dirs_exist_ok=True)

    # Copy requirements.txt to root
    req_src = PROJECT_DIR / "requirements.txt"
    if req_src.exists():
        shutil.copy2(req_src, Path(tmpdir) / "requirements.txt")

    # Git operations
    run_cmd(f'git config user.email "{GITHUB_USERNAME}@users.noreply.github.com"', cwd=tmpdir)
    run_cmd(f'git config user.name "{GITHUB_USERNAME}"', cwd=tmpdir)
    run_cmd("git add -A", cwd=tmpdir)

    # Check if there are changes to commit
    status = run_cmd("git status --porcelain", cwd=tmpdir)
    if status:
        run_cmd('git commit --no-gpg-sign -m "Update pipeline files"', cwd=tmpdir)
        run_cmd("git push origin HEAD", cwd=tmpdir)
        print(f"\nPushed to: https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}")
    else:
        print("No changes to push")

print("Done!")
