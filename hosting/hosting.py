from huggingface_hub import HfApi, create_repo
import os 

api = HfApi(token=os.getenv("HF_TOKEN"))

# Define the repository ID for your Hugging Face Space
# This should be your username followed by the space name, e.g., "your_username/your_space_name"
# Based on the previous actions, 'Tulsi10' is the username. Let's create a new space called 'tourism-app'
repo_id = "Tulsi10/tourism-app"

# Define the local folder containing your Streamlit app files (Dockerfile, app.py, requirements.txt)
folder_to_upload = "Tourism_Project/deployment"

try:
    # Check if the space already exists
    api.repo_info(repo_id=repo_id, repo_type="space")
    print(f"Space '{repo_id}' already exists. Updating files.")
except Exception:
    # If not, create the space
    print(f"Space '{repo_id}' not found. Creating new space.")
    create_repo(repo_id=repo_id, repo_type="space", private=False, space_sdk="docker") # Changed space_sdk to 'docker'
    print(f"Space '{repo_id}' created.")

# Upload the entire folder to the Hugging Face Space
api.upload_folder(
    folder_path=folder_to_upload,
    repo_id=repo_id,
    repo_type="space",                      
    path_in_repo="",                          
)

print(f"Deployment folder '{folder_to_upload}' uploaded to Hugging Face Space '{repo_id}' successfully!")
