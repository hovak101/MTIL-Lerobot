import argparse
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Download a LeRobot dataset from Hugging Face to your local cache.")
    parser.add_argument("--user", required=True, help="Your Hugging Face username")
    parser.add_argument("--repo", required=True, help="The dataset repository name")
    parser.add_argument("--dir", required=True, help="The dataset directory")
    
    args = parser.parse_args()
    
    # Combine user and repo into the format Hugging Face expects
    repo_id = f"{args.user}/{args.repo}"
    
    print(f"Fetching '{repo_id}' from Hugging Face...")
    
    # This automatically fetches the dataset from the Hugging Face Hub 
    # and restores it to your local cache.
    dataset = LeRobotDataset(repo_id, args.dir)
    
    print(f"Success! Downloaded {dataset.num_episodes} episodes to your local cache.")

if __name__ == "__main__":
    main()