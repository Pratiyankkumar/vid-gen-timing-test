import modal
import os
import subprocess
import time
import shutil
from pathlib import Path
import json

# Define environment variables
gpu_env = {
    "MANIMGL_ENABLE_GPU": "1",  # Enable GPU rendering if available
}

cpu_env = {
    "MANIMGL_ENABLE_GPU": "0",  # Disable GPU for CPU-only containers
    "MANIMGL_MULTIPROCESSING": "1",  # Enable multiprocessing for CPU rendering
}

# Create Modal image with required dependencies
image = (
    modal.Image.from_registry("python:3.11-bullseye")
    .run_commands(
        "export DEBIAN_FRONTEND=noninteractive",
        "ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime",
        "apt-get update && apt-get install -y tzdata",
        "dpkg-reconfigure --frontend noninteractive tzdata",
    )
    .apt_install(
        "ffmpeg",
        "build-essential",
        "pkg-config",
        "python3-dev",
        "libgl1-mesa-dev",
        "libegl1-mesa-dev",
        "libgles2-mesa-dev",
        "libglvnd-dev",
        "libglfw3-dev",
        "freeglut3-dev",
        "xvfb",
        "x11-utils",
        "mesa-utils",
        "libcairo2-dev",
        "libpango1.0-dev",
        "sox",
        "libsox-fmt-all",
        "texlive",
        "texlive-latex-extra",
        "texlive-fonts-extra",
        "texlive-latex-recommended",
        "texlive-science",
        "texlive-fonts-recommended",
    )
    .pip_install(
        "manim",
        "numpy",
        "manimpango",
        "requests",
        "pycairo",
        "pyglet",
        "pydub",
        "moderngl",
        "moderngl-window",
        "screeninfo",
        "mapbox-earcut",
        "validators",
        "tqdm",
        "pandas",
        "matplotlib",
        "ffmpeg-python",
    )
    .add_local_dir(".", remote_path="/root/local")
)

app = modal.App("manim-multi-class-rendering", image=image)
# Create a single shared volume for all files
volume = modal.Volume.from_name("manim-outputs-multi-class", create_if_missing=True)

# Helper function to check for videos and list directories
def list_directory_contents(path, depth=0, max_depth=3):
    """List contents of a directory recursively up to max_depth"""
    result = []
    if depth > max_depth:
        return result
    
    if not os.path.exists(path):
        return [f"Directory does not exist: {path}"]
    
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            result.append(f"{'  ' * depth}[DIR] {item}")
            result.extend(list_directory_contents(full_path, depth + 1, max_depth))
        else:
            size = os.path.getsize(full_path)
            result.append(f"{'  ' * depth}[{size} bytes] {item}")
    
    return result

# Function to upload a script to the Modal environment
@app.function(volumes={"/root/shared": volume})
def upload_script(script_content, script_name):
    """Upload a script to the shared volume with better error handling"""
    script_path = f"/root/shared/{script_name}"
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Verify the file was actually written
    if os.path.exists(script_path):
        file_size = os.path.getsize(script_path)
        print(f"Script uploaded to: {script_path} ({file_size} bytes)")
        
        # For debugging, show the first few lines
        with open(script_path, "r") as f:
            head = "".join(f.readlines()[:5])
        print(f"File starts with: {head}")
        
        return script_path
    else:
        print(f"ERROR: Failed to upload script to {script_path}")
        return None

# Function to check if a file exists and copy from local if needed
@app.function(volumes={"/root/shared": volume})
def check_and_fix_script_path(script_path):
    """
    Check if the script exists in the shared volume and copy from local if needed.
    Returns the correct path to use.
    """
    if os.path.exists(script_path):
        print(f"Script found at: {script_path}")
        return script_path
    
    # If not found in shared volume, check local directory
    local_script_path = f"/root/local/{os.path.basename(script_path)}"
    if os.path.exists(local_script_path):
        print(f"Script found in local directory: {local_script_path}")
        
        # Copy it to the shared volume for consistency
        shared_path = f"/root/shared/{os.path.basename(script_path)}"
        os.makedirs(os.path.dirname(shared_path), exist_ok=True)
        shutil.copy2(local_script_path, shared_path)
        print(f"Copied script from {local_script_path} to {shared_path}")
        
        return shared_path
    
    # Neither location has the file
    print(f"ERROR: Script not found in either location!")
    
    # List directories to help debug
    contents = list_directory_contents("/root/shared")
    print(f"Contents of /root/shared:")
    for line in contents:
        print(line)
    
    contents = list_directory_contents("/root/local")
    print(f"Contents of /root/local:")
    for line in contents:
        print(line)
    
    return None

def generate_class_render_script(original_script_path, class_name):
    """
    Generate a modified Manim script that will only render a specific animation class.
    """
    # Read the original script
    with open(original_script_path, "r") as f:
        script_content = f.read()
    
    # Create a class-specific script
    class_script = f"""
# Modified script to render only the {class_name} class
import sys
from manim import *

# Original script content
{script_content}

if __name__ == "__main__":
    # Use standard resolution and frame rate
    config.pixel_height = 720
    config.pixel_width = 1280
    config.frame_rate = 30
    {class_name}().render()
"""

    # Write the modified script to a temporary file
    class_script_path = f"/tmp/{class_name}_render.py"
    with open(class_script_path, "w") as f:
        f.write(class_script)
    
    return class_script_path

@app.function(
    volumes={"/root/shared": volume},
    timeout=1200,
    gpu="any" if os.environ.get("USE_GPU", "0") == "1" else None,
    cpu=4 if os.environ.get("USE_GPU", "0") == "0" else 2,
    memory=8192 if os.environ.get("USE_GPU", "0") == "0" else 4096,
)
def render_animation_class(script_path, class_name):
    """Render a specific animation class using Manim"""
    # Set up environment variables based on hardware
    has_gpu = os.environ.get("MODAL_CONTAINER_GPU_COUNT", "0") != "0"
    env_vars = gpu_env if has_gpu else cpu_env
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Configure output paths
    output_dir = f"/root/shared/output/class_{class_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up virtual display
    os.environ["DISPLAY"] = ":1"
    display_process = subprocess.Popen(["Xvfb", ":1", "-screen", "0", "1920x1080x24"])
    time.sleep(2)
    
    try:
        # Generate a modified script for this class
        class_script_path = generate_class_render_script(script_path, class_name)
        print(f"Generated class script at: {class_script_path}")
        
        # Render the class
        start_time = time.time()
        output_path = f"{output_dir}/{class_name}.mp4"
        
        # Run the script directly with Python
        cmd = f"python {class_script_path}"
        print(f"Running command: {cmd}")
        
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Stdout: {process.stdout}")
        print(f"Stderr: {process.stderr}")
        
        render_time = time.time() - start_time
        
        # Find the rendered output file
        found_mp4_files = []
        search_dirs = ["/tmp", "./media", os.path.dirname(class_script_path)]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith(".mp4") and class_name in file:
                            found_mp4_files.append(os.path.join(root, file))
        
        if found_mp4_files:
            # Copy to output location
            class_output_path = os.path.join(output_dir, f"{class_name}.mp4")
            shutil.copy2(found_mp4_files[0], class_output_path)
            print(f"Copied {class_name} to {class_output_path}")
            
            return {
                "class_name": class_name,
                "success": True,
                "render_time": render_time,
                "output_file": class_output_path,
            }
        else:
            print(f"No output file found for class {class_name}")
            return {
                "class_name": class_name,
                "success": False,
                "render_time": render_time,
                "output_file": None,
                "error": "No output file found"
            }
    finally:
        # Clean up the virtual display
        if display_process:
            display_process.terminate()

@app.function(volumes={"/root/shared": volume})
def find_class_files():
    """Find all rendered class files in the output directory - improved version"""
    valid_class_paths = []
    volume_base = "/root/shared/output"
    
    # Make sure the directory exists
    if not os.path.exists(volume_base):
        print(f"Volume base directory {volume_base} does not exist")
        return valid_class_paths
    
    print(f"Scanning all directories under {volume_base} for MP4 files:")
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(volume_base):
        for file in files:
            if file.endswith(".mp4"):
                full_path = os.path.join(root, file)
                file_size = os.path.getsize(full_path)
                if file_size > 0:  # Check for non-empty files
                    print(f"Found MP4: {full_path} ({file_size} bytes)")
                    valid_class_paths.append(full_path)
                else:
                    print(f"Skipping empty MP4: {full_path}")
    
    print(f"Total valid MP4 files found: {len(valid_class_paths)}")
    return valid_class_paths

@app.function(volumes={"/root/shared": volume}, timeout=600)
def concat_videos(video_paths, output_path):
    """Concatenate multiple videos together using ffmpeg"""
    # Create a temporary directory for intermediate files
    temp_dir = "/tmp/merge_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a file list for ffmpeg
    concat_file = f"{temp_dir}/concat_list.txt"
    with open(concat_file, "w") as f:
        for path in video_paths:
            if os.path.exists(path):
                f.write(f"file '{path}'\n")
    
    # Verify concat file contents
    with open(concat_file, "r") as f:
        print(f"Concat file contents:\n{f.read()}")
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use ffmpeg to concatenate videos
    cmd = f"ffmpeg -f concat -safe 0 -i {concat_file} -c:v libx264 -preset medium -crf 22 {output_path}"
    
    print(f"Running command: {cmd}")
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    success = process.returncode == 0
    
    if success:
        print(f"Successfully combined videos into: {output_path}")
        print(f"Output file size: {os.path.getsize(output_path)} bytes")
    else:
        print(f"Failed to combine videos: {process.stderr}")
    
    # Return result
    return {
        "success": success,
        "output_file": output_path if success else None,
        "videos_count": len(video_paths),
        "stdout": process.stdout,
        "stderr": process.stderr,
        "command": cmd
    }

@app.function(volumes={"/root/shared": volume})
def extract_animation_classes(script_path):
    """Extract the names of all animation classes from the script"""
    with open(script_path, "r") as f:
        script_content = f.read()
    
    # Set up virtual display for Manim import
    os.environ["DISPLAY"] = ":1"
    display_process = subprocess.Popen(["Xvfb", ":1", "-screen", "0", "1920x1080x24"])
    time.sleep(2)
    
    try:
        # Write a temporary script to extract the classes
        temp_script_path = "/tmp/extract_classes.py"
        with open(temp_script_path, "w") as f:
            f.write(f"""
import inspect
import sys
import json
from manim import Scene

# Add the directory of the original script to the path
sys.path.append('{os.path.dirname(script_path)}')

# Import the module (the script)
module_name = '{os.path.basename(script_path).replace(".py", "")}'
module = __import__(module_name)

# Find all Scene subclasses
scene_classes = []
for name, obj in inspect.getmembers(module):
    if inspect.isclass(obj) and issubclass(obj, Scene) and obj != Scene:
        scene_classes.append(name)

# Output as JSON
print(json.dumps(scene_classes))
""")
        
        # Run the script to extract classes
        result = subprocess.run(
            ["python", temp_script_path],
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error extracting classes: {result.stderr}")
            # Fallback: Simple regex search for potential class names
            import re
            class_matches = re.findall(r'class\s+(\w+)\((?:Scene|.*?Scene)\):', script_content)
            return class_matches
        
        # Parse the JSON output
        return json.loads(result.stdout)
    finally:
        if display_process:
            display_process.terminate()

@app.function(volumes={"/root/shared": volume})
def list_output_directory():
    output_dir = "/root/shared/output"
    result = []
    
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            full_path = os.path.join(output_dir, item)
            if os.path.isdir(full_path):
                result.append(f"[DIR] {item}")
            else:
                size = os.path.getsize(full_path)
                result.append(f"[{size} bytes] {item}")
    else:
        result.append(f"Directory does not exist: {output_dir}")
    
    return result

@app.local_entrypoint()
def main(
    script_name="binary_search.py",
    max_containers=6,
    use_gpu=False
):
    """Main entry point for parallel class rendering"""
    if use_gpu:
        os.environ["USE_GPU"] = "1"
    
    print(f"Starting Parallel Class Rendering with {'GPU' if use_gpu else 'CPU'} and max {max_containers} containers...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_script_path = os.path.join(script_dir, script_name)
    
    if not os.path.exists(local_script_path):
        print(f"Error: Script {local_script_path} not found!")
        return
    
    # Upload the script to Modal
    print(f"\n=== Uploading Script {script_name} ===")
    with open(local_script_path, "r") as f:
        script_content = f.read()
    
    remote_script_path = upload_script.remote(script_content, script_name)
    verified_script_path = check_and_fix_script_path.remote(remote_script_path)
    
    if not verified_script_path:
        print("ERROR: Failed to upload or locate the script. Exiting.")
        return
    
    # Extract animation classes from the script
    print("\n=== Extracting Animation Classes ===")
    animation_classes = extract_animation_classes.remote(verified_script_path)
    
    if not animation_classes:
        print("ERROR: No animation classes found in the script. Exiting.")
        return
    
    print(f"Found {len(animation_classes)} animation classes:")
    for cls in animation_classes:
        print(f"  - {cls}")
    
    # Limit to max_containers if needed
    if len(animation_classes) > max_containers:
        print(f"Limiting to {max_containers} containers as specified")
        animation_classes = animation_classes[:max_containers]
    
    # Render classes in parallel
    print("\n=== Rendering Animation Classes in Parallel ===")
    start_time = time.time()
    class_results = render_animation_class.map(
        [verified_script_path] * len(animation_classes), 
        animation_classes
    )
    render_time = time.time() - start_time
    
    # Process results
    class_paths = []
    for result in class_results:
        if result["success"]:
            print(f"Class {result['class_name']}: Success ({result['render_time']:.2f}s)")
            class_paths.append(result["output_file"])
        else:
            print(f"Class {result['class_name']}: Failed")
    
    if not class_paths:
        print("No classes were rendered successfully. Exiting.")
        return
    
    # Find all rendered class files
    print("\n=== Finding Rendered Class Files ===")
    valid_class_paths = find_class_files.remote()
    
    if not valid_class_paths:
        print("No valid class files found. Exiting.")
        return
    
    print(f"\nFound {len(valid_class_paths)} valid class renderings:")
    for path in valid_class_paths:
        print(f"  - {path}")
    
    # Concatenate all class videos
    print("\n=== Concatenating Class Videos ===")
    output_filename = f"combined_animations.mp4"
    output_path = f"/root/shared/output/{output_filename}"
    
    concat_result = concat_videos.remote(valid_class_paths, output_path)
    
    if concat_result["success"]:
        print(f"Successfully concatenated videos into: {concat_result['output_file']}")
        
        # List the output directory to confirm the file exists
        output_contents = list_output_directory.remote()
        print("\n=== Output Directory Contents ===")
        for line in output_contents:
            print(line)
            
        print(f"\nRendering complete! Final output is stored in the volume at: {output_path}")
    else:
        print("Failed to concatenate videos. Details:")
        print(f"Command used: {concat_result['command']}")
        print(f"Error: {concat_result.get('stderr', 'No error details available')}")