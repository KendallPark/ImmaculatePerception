import argparse
import os
# Disable HF Transfer to avoid MerkleDB/Rust-based crashes in this environment
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import HfApi, create_repo, whoami
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

MODEL_CARD_TEMPLATE = """
---
library_name: maryvlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
---

**MaryVLM** is a minimal and lightweight Vision-Language Model (VLM) used for the MaryVLM project.
"""

def main():
    parser = argparse.ArgumentParser(description="Push a trained checkpoint to Hugging Face Hub")
    parser.add_argument("checkpoint_path", type=str, help="Path to the local checkpoint directory")
    parser.add_argument("repo_id", type=str, nargs="?", help="Target Hugging Face Hub Repo ID. Auto-generated if empty.")
    parser.add_argument("--private", action="store_true", help="Make the repository private")

    args = parser.parse_args()

    # Resolve Repo ID
    repo_id = args.repo_id
    if not repo_id:
        try:
            user_info = whoami()
            username = user_info['name']
        except Exception as e:
            print("Could not detect Hugging Face username. Please login or provide repo_id.")
            return

        path = args.checkpoint_path.rstrip(os.sep)
        if "checkpoint-" in os.path.basename(path):
            model_name = os.path.basename(os.path.dirname(path))
        else:
            model_name = os.path.basename(path)
        repo_id = f"{username}/{model_name}"
        print(f"Auto-detected repo_id: {repo_id}")

    print(f"Preparing to push {args.checkpoint_path} to {repo_id}...")

    # Extract Metadata
    checkpoint_num = "Unknown"
    basename = os.path.basename(args.checkpoint_path.rstrip(os.sep))
    if "checkpoint-" in basename:
        try:
            checkpoint_num = basename.split("-")[-1]
        except:
            pass

    # Create Repo
    try:
        url = create_repo(repo_id=repo_id, private=args.private, exist_ok=True)
        print(f"Repo ready: {url}")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # Check/Create README in the directory
    readme_path = os.path.join(args.checkpoint_path, "README.md")
    if not os.path.exists(readme_path):
        print("Generating README.md...")
        content = MODEL_CARD_TEMPLATE
        content += f"\n\n## Training Metadata\n- **Checkpoint**: {checkpoint_num}\n"
        with open(readme_path, "w") as f:
            f.write(content)
    # Note: we are not appending if it exists to avoid infinite growth on retries for this simplistic script

    # Upload Files Individually (More robust than upload_folder in some environments)
    print("Uploading files individually...")
    api = HfApi()

    files_to_upload = []
    for root, _, files in os.walk(args.checkpoint_path):
        for file in files:
            if file in ["optimizer.pt", "rng_state.pth", "scheduler.pt", "training_args.bin"]:
                continue
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, args.checkpoint_path)
            files_to_upload.append((abs_path, rel_path))

    for abs_path, rel_path in files_to_upload:
        print(f"Uploading {rel_path}...")
        try:
            api.upload_file(
                path_or_fileobj=abs_path,
                path_in_repo=rel_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {rel_path} (checkpoint {checkpoint_num})"
            )
        except Exception as e:
            print(f"Failed to upload {rel_path}: {e}")
            # Continue trying generic files even if one fails

    print(f"Done! Model: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
