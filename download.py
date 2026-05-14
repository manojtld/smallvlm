import kagglehub

# Download latest version
# download to a specified path, or to the current working directory if not specified
output_path = "/raid/manoj/smallvlm/"
path = kagglehub.dataset_download("raddar/chest-xrays-indiana-university")

print("Path to dataset files:", path)