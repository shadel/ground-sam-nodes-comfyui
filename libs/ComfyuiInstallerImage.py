import requests
import json
from server import PromptServer
import folder_paths
import os
import git

def get_default_clone_folder(github_url):
    # Split the URL to get the repository name
    repo_name_with_git = github_url.rstrip('/').split('/')[-1]
    
    # Remove the .git extension if it's there
    repo_name = repo_name_with_git[:-4] if repo_name_with_git.endswith('.git') else repo_name
    
    return repo_name

def get_latest_commit_hash(repo_folder):
    try:
        # Initialize the repo object
        repo = git.Repo(repo_folder)
        
        # Get the latest commit hash from the current branch
        commit_hash = repo.head.commit.hexsha
        
        return commit_hash
    except Exception as e:
        print(f"Error accessing the repository: {e}")
        return None

class ComfyuiInstallerImage:
    def __init__(self):
        self.button_text = "Fetch Data"
        self.api_url = "http://localhost:8188/api/customnode/getlist?mode=cache&skip_update=true"  # Replace with your actual API URL
        self.data = None

    def get_custom_list_data(self):
        # Display the retrieved data
        try:
            # Make the API request
            response = requests.get(self.api_url)
            if response.status_code == 200:
                self.data = response.json()
            else:
                print(f"API request failed with status code: {response.status_code}")
        except Exception as e:
            print(f"Error while making API request: {e}")

        return self.data

    def get_layer(self):
        custom_list_data = self.get_custom_list_data()
        custom_nodes = custom_list_data['custom_nodes']

        installed_nodes = []
        for node in custom_nodes:
            if node["installed"] == "True":
                installed_nodes.append(node)

        
        comfy_ui_commit_hash = get_latest_commit_hash(folder_paths.base_path)

        custom_nodes_with_commit_hash = []

        for node in installed_nodes:
            node_version = self.get_node_with_commithash(node)
            print(node_version)
            custom_nodes_with_commit_hash.append(node_version)

        return {
            "comfy_ui_commit_hash": comfy_ui_commit_hash,
            "custom_nodes": custom_nodes_with_commit_hash
        }


    def get_node_with_commithash(self, node):
        file_url = node["files"][0]
        commit_hash = self.get_node_commithash(node)

        return {
            "title": node["title"],
            "id": node["id"],
            "url": file_url,
            "commit_hash": commit_hash
        }

    def get_node_commithash(self, node):
        file_url = node["files"][0]
        folder_name = get_default_clone_folder(file_url)
        return get_latest_commit_hash(os.path.join(folder_paths.base_path, "custom_nodes", folder_name))
        


from aiohttp import web

install_image = ComfyuiInstallerImage()
    
@PromptServer.instance.routes.get("/install-layer/get-layer")
async def get_layers(request):

    json_obj = install_image.get_layer()

    return web.json_response(json_obj, content_type='application/json')