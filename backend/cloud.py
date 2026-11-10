"""Cloud projects: save studio projects to the signed-in user's own Hugging Face account.

Each user gets a private dataset ``<username>/musiclab-projects``; every project is one
``projects/<name>.json`` file in it (the same format as 「保存工程」, audio included).
The token comes from "Sign in with Hugging Face" on the Space, so nothing is stored here.
"""

import re
from pathlib import Path

REPO_NAME = "musiclab-projects"


def _api(api=None):
    if api is not None:
        return api
    from huggingface_hub import HfApi

    return HfApi()


def safe_name(name: str) -> str:
    s = re.sub(r'[\\/:*?"<>|#%\s]+', "_", (name or "").strip()).strip("._")
    return s[:80] or "untitled"


def repo_id(token: str, api=None) -> str:
    return f"{_api(api).whoami(token=token)['name']}/{REPO_NAME}"


def list_projects(token: str, api=None) -> dict:
    api = _api(api)
    rid = repo_id(token, api)
    if not api.repo_exists(rid, repo_type="dataset", token=token):
        return {"repo": rid, "projects": []}
    files = api.list_repo_files(rid, repo_type="dataset", token=token)
    names = sorted(f[len("projects/"):-len(".json")] for f in files if f.startswith("projects/") and f.endswith(".json"))
    return {"repo": rid, "projects": names}


def save_project(token: str, name: str, path: Path, api=None) -> dict:
    api = _api(api)
    rid = repo_id(token, api)
    api.create_repo(rid, repo_type="dataset", private=True, exist_ok=True, token=token)
    fn = safe_name(name)
    api.upload_file(path_or_fileobj=str(path), path_in_repo=f"projects/{fn}.json", repo_id=rid,
                    repo_type="dataset", token=token, commit_message=f"保存工程 {fn}")
    return {"repo": rid, "name": fn}


def load_project(token: str, name: str, api=None) -> str:
    api = _api(api)
    return api.hf_hub_download(repo_id(token, api), f"projects/{safe_name(name)}.json", repo_type="dataset", token=token)
