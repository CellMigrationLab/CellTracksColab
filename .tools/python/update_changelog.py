#!/usr/bin/env python3
"""
Script to automatically update changelog files when versions change in latest_version.yaml
"""

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import yaml

def get_git_commit_info():
    """Get current commit hash and message"""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        commit_message = subprocess.check_output(['git', 'log', '-1', '--pretty=%B']).decode('utf-8').strip()
        commit_author = subprocess.check_output(['git', 'log', '-1', '--pretty=%an']).decode('utf-8').strip()
        commit_date = subprocess.check_output(['git', 'log', '-1', '--pretty=%ai']).decode('utf-8').strip()
        
        # Get the remote origin URL to construct GitHub link
        try:
            remote_url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url']).decode('utf-8').strip()
            # Convert SSH or HTTPS URL to GitHub web URL
            if remote_url.startswith('git@github.com:'):
                # SSH format: git@github.com:owner/repo.git
                repo_path = remote_url.replace('git@github.com:', '').replace('.git', '')
            elif 'github.com' in remote_url:
                # HTTPS format: https://github.com/owner/repo.git
                repo_path = remote_url.split('github.com/')[-1].replace('.git', '')
            else:
                repo_path = None
            
            github_commit_url = f"https://github.com/{repo_path}/commit/{commit_hash}" if repo_path else None
        except subprocess.CalledProcessError:
            github_commit_url = None
        
        return {
            'hash': commit_hash,
            'message': commit_message,
            'author': commit_author,
            'date': commit_date,
            'github_url': github_commit_url
        }
    except subprocess.CalledProcessError as e:
        print(f"Error getting git info: {e}")
        return None

def flatten_versions(data) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Flatten possibly nested version mappings into notebook -> version."""
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


def get_previous_version_file(version_file_path):
    """Get the previous version of the version manifest from git"""
    try:
        previous_content = subprocess.check_output(
            ['git', 'show', f'HEAD~1:{version_file_path}'],
            text=True,
        )
        data = yaml.safe_load(previous_content) or {}
        return flatten_versions(data)
    except subprocess.CalledProcessError:
        return {}, {}

def load_current_versions(version_file_path):
    """Load current versions from the manifest file."""
    version_file = Path(version_file_path)
    if version_file.exists():
        with open(version_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            return flatten_versions(data)
    return {}, {}


def find_notebook_folder(root_path, project_name):
    """Find the corresponding notebook folder for a project."""
    root_path = Path(root_path)

    for folder in root_path.iterdir():
        if folder.name == '__pycache__':
            continue
        if folder.is_dir():
            notebook_path = find_notebook_folder(folder, project_name)
            if notebook_path:
                return notebook_path
        elif folder.stem == project_name and folder.suffix == '.ipynb':
            return folder.parent
        
    return None


def create_or_update_changelog(project_name, old_version, new_version, commit_info, version_file_path, category=None):
    """Create or update changelog file for a project."""
    folder_path = find_notebook_folder(Path(version_file_path).parent, project_name)
    if folder_path is None:
        print(f"Skipping changelog update for {project_name}: notebook folder not found.")
        return

    changelog_path = folder_path / 'CHANGELOG.md'
    
    # Create changelog entry
    entry_date = datetime.now().strftime('%Y-%m-%d')
    
    # Create hash field with link if GitHub URL is available
    if commit_info.get('github_url'):
        hash_field = f"[`{commit_info['hash'][:8]}`]({commit_info['github_url']})"
    else:
        hash_field = f"`{commit_info['hash'][:8]}`"
    
    changelog_entry = f"""
## [{new_version}] - {entry_date}

### Changed
- Version updated from {old_version} to {new_version}

**Commit Details:**
- Hash: {hash_field}
- Author: {commit_info['author']}
- Date: {commit_info['date']}
- Message: {commit_info['message']}

---
"""

    if changelog_path.exists():
        # Read existing changelog
        with open(changelog_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find where to insert the new entry (after the title)
        lines = content.split('\n')
        insert_index = 0
        
        # Look for the first ## heading or insert after title
        for i, line in enumerate(lines):
            if line.startswith('# '):
                insert_index = i + 1
                # Skip any description lines until we find the first version entry
                while insert_index < len(lines) and not lines[insert_index].startswith('## ['):
                    insert_index += 1
                break
        
        # Insert the new entry
        lines.insert(insert_index, changelog_entry.strip())
        new_content = '\n'.join(lines)
        
    else:
        # Create new changelog
        new_content = f"""# Changelog - {project_name}

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

{changelog_entry.strip()}
"""
    
    # Write the updated changelog
    with open(changelog_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Updated changelog for {project_name}: {changelog_path}")

def main():
    """Main function to check for version changes and update changelogs"""
    parser = argparse.ArgumentParser(description='Update changelog files when versions change')
    parser.add_argument('version_file', help='Path to the latest_version.yaml file that changed')
    
    args = parser.parse_args()
    version_file_path = args.version_file
    
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
    
    # Check for changes
    changes_found = False
    for project_name, current_version in current_versions.items():
        previous_version = previous_versions.get(project_name, "0.0.0")
        if current_version != previous_version:
            print(f"Version change detected for {project_name}: {previous_version} -> {current_version}")
            category = category_map.get(project_name)
            create_or_update_changelog(
                project_name,
                previous_version,
                current_version,
                commit_info,
                version_file_path,
                category=category,
            )
            changes_found = True
            continue

    # Detect removed entries (optional)
    removed_projects = set(previous_versions) - set(current_versions)
    if removed_projects:
        print(f"Projects removed from manifest: {', '.join(sorted(removed_projects))}")
    
    if not changes_found:
        print("No version changes detected.")
    else:
        print("Changelog updates completed.")

if __name__ == "__main__":
    main()
