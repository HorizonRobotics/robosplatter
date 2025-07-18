from huggingface_hub import snapshot_download

snapshot_download(repo_id="HorizonRobotics/RoboSplatter", 
    local_dir='.', 
    repo_type="dataset",
    resume_download=True)