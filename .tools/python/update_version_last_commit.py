import argparse
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import yaml

def get_git_commit_info():
    """Get current commit hash, parent hash, and message"""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        parent_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD^']).decode('utf-8').strip()
        
        # Get the remote origin URL to construct GitHub link
        try:
            remote_url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url']).decode('utf-8').strip()
            # Convert SSH or HTTPS URL to GitHub web URL
            if remote_url.startswith('git@github.com:'):
                repo_path = remote_url.replace('git@github.com:', '').replace('.git', '')
            elif 'github.com' in remote_url:
                repo_path = remote_url.split('github.com/')[-1].replace('.git', '')
            else:
                repo_path = None
            
            github_commit_url = f"https://github.com/{repo_path}/commit/{commit_hash}" if repo_path else None
            github_parent_url = f"https://github.com/{repo_path}/commit/{parent_hash}" if repo_path else None
        except subprocess.CalledProcessError:
            github_commit_url = None
            github_parent_url = None
        
        return {
            'hash': commit_hash,
            'parent_hash': parent_hash,
            'github_url': github_commit_url,
            'github_parent_url': github_parent_url
        }
    except subprocess.CalledProcessError as e:
        print(f"Error getting git info: {e}")

def flatten_versions(data) -> Tuple[Dict[str, str], Dict[str, str]]:
    versions: Dict[str, str] = {}
    categories: Dict[str, str] = {}

    if not data:
        return versions, categories

    for key, value in data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                versions[sub_key] = sub_value
                categories[sub_key] = key
        else:
            versions[key] = value
            categories.setdefault(key, None)
    return versions, categories


def load_current_versions(version_file_path):
    """Load current versions from the manifest."""
    version_file = Path(version_file_path)
    if version_file.exists():
        with open(version_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            return flatten_versions(data)
    return {}, {}

def get_previous_version_file(version_file_path):
    """Get the previous version of the manifest from git."""
    try:
        previous_content = subprocess.check_output(
            ['git', 'show', f'HEAD~1:{version_file_path}'],
            text=True,
        )
        data = yaml.safe_load(previous_content) or {}
        return flatten_versions(data)
    except subprocess.CalledProcessError:
        return {}, {}


def find_notebook_folder(project_name, version_file_path, category=None):
    """Locate the notebook directory for a project."""
    version_dir = Path(version_file_path).parent
    if category:
        project_folder = version_dir / category / project_name
        if project_folder.exists():
            return project_folder

    for folder in version_dir.iterdir():
        if not folder.is_dir() or folder.name == '__pycache__':
            continue
        project_folder = folder / project_name
        if project_folder.exists():
            return project_folder
    return None

def main():
    """Main function to check for version changes and update changelogs"""
    parser = argparse.ArgumentParser(description='Update changelog files when versions change')
    parser.add_argument('version_file', help='Path to the latest_version.yaml file that changed')
    
    args = parser.parse_args()
    version_file_path = Path(args.version_file)
    
    print(f"Processing version file: {version_file_path}")
    
    # Get git commit information
    commit_info = get_git_commit_info()
    if not commit_info:
        print("Error: Could not get git commit information")
        return

    # Load current and previous versions
    current_versions, category_map = load_current_versions(version_file_path)
    previous_versions, _ = get_previous_version_file(version_file_path)
    
    print(f"Current versions: {current_versions}")
    print(f"Previous versions: {previous_versions}")

    changes_found = False
    for project_name, current_version in current_versions.items():
        previous_version = previous_versions.get(project_name, "0.0.0")
        if current_version != previous_version:
            print(f"Version change detected for {project_name}: {previous_version} -> {current_version}")

            folder = find_notebook_folder(project_name, version_file_path, category_map.get(project_name))
            if folder is None:
                print(f"Skipping last-commit update for {project_name}: notebook folder not found.")
                continue

            version_last_config_path = folder / "version_last_commit.yaml"

            # Load last commit info if it exists
            last_commit_info = {}
            if version_last_config_path.exists():
                with open(version_last_config_path, 'r', encoding='utf-8') as f:
                    last_commit_info = yaml.safe_load(f) or {}
            
            # Add the project into the last commit info
            if project_name not in last_commit_info:
                last_commit_info[project_name] = {previous_version: commit_info['parent_hash']}
            else:
                last_commit_info[project_name][previous_version] = commit_info['parent_hash']

            with open(version_last_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(last_commit_info, f)

            changes_found = True

    if not changes_found:
        print("No version changes detected.")
    else:
        print(f"Version's last commits updated on {version_file_path}")

if __name__ == "__main__":
    main()
