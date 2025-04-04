import modal
import os
import subprocess
import time
import shutil
from pathlib import Path
import json
import datetime

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

# Improved helper function to check for videos and list directories
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

# Improved function to upload a script to the Modal environment with better verification
@app.function(volumes={"/root/shared": volume})
def upload_script(script_content, script_name):
    """Upload a script to the shared volume with robust verification"""
    # Create directory structure if it doesn't exist
    script_dir = "/root/shared"
    os.makedirs(script_dir, exist_ok=True)
    script_path = f"{script_dir}/{script_name}"
    
    print(f"Writing script to {script_path}...")
    
    # Write the script with explicit flush
    with open(script_path, "w") as f:
        f.write(script_content)
        f.flush()
        os.fsync(f.fileno())  # Force sync to disk
    
    # Verify the file exists and has content
    if os.path.exists(script_path) and os.path.getsize(script_path) > 0:
        file_size = os.path.getsize(script_path)
        print(f"Successfully wrote script to: {script_path} ({file_size} bytes)")
        
        # For debugging, show the first and last few lines
        with open(script_path, "r") as f:
            lines = f.readlines()
            head = "".join(lines[:5])
            tail = "".join(lines[-5:]) if len(lines) > 5 else ""
        
        print(f"File starts with: {head}")
        print(f"File ends with: {tail}")
        
        # List directory contents to confirm
        print("Directory contents after writing:")
        dir_contents = os.listdir(script_dir)
        for item in dir_contents:
            item_path = os.path.join(script_dir, item)
            size = os.path.getsize(item_path) if os.path.isfile(item_path) else "<DIR>"
            print(f"  - {item} ({size})")
            
        return script_path
    else:
        print(f"ERROR: Failed to write script to {script_path}")
        return None

# Improved function to extract animation classes using script content directly
@app.function(volumes={"/root/shared": volume})
def extract_animation_classes_from_content(script_content, script_name):
    """Extract animation classes directly from script content to avoid volume sync issues"""
    print(f"Extracting animation classes from script content for {script_name}")
    
    # First save the script to a temporary location
    temp_script_path = f"/tmp/{script_name}"
    with open(temp_script_path, "w") as f:
        f.write(script_content)
    
    # Set up virtual display for Manim import
    os.environ["DISPLAY"] = ":1"
    display_process = subprocess.Popen(["Xvfb", ":1", "-screen", "0", "1920x1080x24"])
    time.sleep(2)
    
    try:
        # Write an extraction script
        extract_script_path = "/tmp/extract_classes.py"
        with open(extract_script_path, "w") as f:
            f.write(f"""
import os
import sys
import json
import importlib.util
from manim import Scene

# Define the path to the temporary script
script_path = "{temp_script_path}"

# Load the module dynamically
spec = importlib.util.spec_from_file_location("dynamic_module", script_path)
module = importlib.util.module_from_spec(spec)
sys.modules["dynamic_module"] = module
spec.loader.exec_module(module)

# Find all Scene subclasses
scene_classes = []
for name in dir(module):
    obj = getattr(module, name)
    try:
        if isinstance(obj, type) and issubclass(obj, Scene) and obj != Scene:
            scene_classes.append(name)
    except TypeError:
        # This happens when obj is not a class
        pass

# Output as JSON
print(json.dumps(scene_classes))
""")
        
        # Execute the extraction script
        result = subprocess.run(
            ["python", extract_script_path],
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error extracting classes: {result.stderr}")
            # Fallback: Simple regex search for potential class names
            import re
            class_matches = re.findall(r'class\s+(\w+)\((?:Scene|.*?Scene)\):', script_content)
            print(f"Using regex fallback, found classes: {class_matches}")
            return class_matches
        
        # Parse the JSON output
        classes = json.loads(result.stdout)
        print(f"Successfully extracted {len(classes)} animation classes: {classes}")
        return classes
    finally:
        if display_process:
            display_process.terminate()

def generate_class_render_script(script_content, class_name):
    """
    Generate a modified Manim script that will only render a specific animation class.
    Works with script content directly rather than a file path.
    """
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
def render_animation_class(script_content, class_name):
    """Render a specific animation class using Manim, taking script content directly"""
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
        class_script_path = generate_class_render_script(script_content, class_name)
        print(f"Generated class script at: {class_script_path}")
        
        # Verify the script was created correctly
        if os.path.exists(class_script_path):
            print(f"Class script exists: {class_script_path} ({os.path.getsize(class_script_path)} bytes)")
        else:
            print(f"ERROR: Class script not found at {class_script_path}")
            return {
                "class_name": class_name,
                "success": False,
                "error": "Failed to create class render script"
            }
        
        # Render the class
        start_time = time.time()
        output_path = f"{output_dir}/{class_name}.mp4"
        
        # Run the script directly with Python
        cmd = f"python {class_script_path}"
        print(f"Running command: {cmd}")
        
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Command exit code: {process.returncode}")
        print(f"Stdout: {process.stdout}")
        print(f"Stderr: {process.stderr}")
        
        render_time = time.time() - start_time
        
        # Find the rendered output file
        found_mp4_files = []
        search_dirs = ["/tmp", "./media", os.path.dirname(class_script_path)]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                print(f"Searching for MP4 files in {search_dir}")
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith(".mp4") and class_name in file:
                            mp4_path = os.path.join(root, file)
                            found_mp4_files.append(mp4_path)
                            print(f"Found MP4: {mp4_path} ({os.path.getsize(mp4_path)} bytes)")
        
        if found_mp4_files:
            # Copy to output location
            class_output_path = os.path.join(output_dir, f"{class_name}.mp4")
            shutil.copy2(found_mp4_files[0], class_output_path)
            print(f"Copied {class_name} to {class_output_path}")
            
            # Verify the copy succeeded
            if os.path.exists(class_output_path) and os.path.getsize(class_output_path) > 0:
                print(f"Verified output file: {class_output_path} ({os.path.getsize(class_output_path)} bytes)")
            else:
                print(f"WARNING: Copy verification failed for {class_output_path}")
            
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
    
    # Verify all input files exist and have content
    valid_paths = []
    print(f"Validating {len(video_paths)} video paths...")
    for path in video_paths:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            valid_paths.append(path)
            print(f"Valid file: {path} ({os.path.getsize(path)} bytes)")
        else:
            print(f"Warning: Invalid or empty file: {path}")
    
    if not valid_paths:
        print("No valid video files to concatenate!")
        return {
            "success": False,
            "error": "No valid video files found"
        }
    
    # Create a file list for ffmpeg
    concat_file = f"{temp_dir}/concat_list.txt"
    with open(concat_file, "w") as f:
        for path in valid_paths:
            f.write(f"file '{path}'\n")
    
    # Verify concat file contents
    with open(concat_file, "r") as f:
        concat_content = f.read()
        print(f"Concat file contents:\n{concat_content}")
    
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
        "videos_count": len(valid_paths),
        "stdout": process.stdout,
        "stderr": process.stderr,
        "command": cmd
    }

@app.function(volumes={"/root/shared": volume})
def list_output_directory():
    output_dir = "/root/shared/output"
    result = []
    
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            rel_path = os.path.relpath(root, output_dir)
            if rel_path == ".":
                prefix = ""
            else:
                prefix = rel_path + "/"
                
            for file in files:
                full_path = os.path.join(root, file)
                size = os.path.getsize(full_path)
                result.append(f"{prefix}{file} ({size} bytes)")
    else:
        result.append(f"Directory does not exist: {output_dir}")
    
    return result

@app.local_entrypoint()
def main(
    script_name="binary_search.py",
    max_containers=6,
    use_gpu=False
):
    """Main entry point for parallel class rendering with improved robustness"""
    if use_gpu:
        os.environ["USE_GPU"] = "1"
    
    # Record start time for total time calculation
    total_start_time = time.time()
    
    print(f"Starting Parallel Class Rendering with {'GPU' if use_gpu else 'CPU'} and max {max_containers} containers...")
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_script_path = os.path.join(script_dir, script_name)
    
    if not os.path.exists(local_script_path):
        print(f"Error: Script {local_script_path} not found!")
        return
    
    # Read script content once
    print(f"\n=== Reading Local Script {script_name} ===")
    with open(local_script_path, "r") as f:
        script_content = f.read()
    
    script_size = len(script_content)
    print(f"Successfully read script: {local_script_path} ({script_size} bytes)")
    
    # Upload the script to Modal (for future use, but we'll primarily use content directly)
    print(f"\n=== Uploading Script {script_name} to Volume ===")
    remote_script_path = upload_script.remote(script_content, script_name)
    
    # Extract animation classes directly from content
    print("\n=== Extracting Animation Classes ===")
    animation_classes = extract_animation_classes_from_content.remote(script_content, script_name)
    
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
    
    # Render classes in parallel - passing script content directly
    print("\n=== Rendering Animation Classes in Parallel ===")
    render_start_time = time.time()
    class_results = render_animation_class.map(
        [script_content] * len(animation_classes), 
        animation_classes
    )
    render_time = time.time() - render_start_time
    
    # Process results
    successful_renders = 0
    class_paths = []
    total_class_render_time = 0
    
    for result in class_results:
        if result["success"]:
            successful_renders += 1
            print(f"Class {result['class_name']}: Success ({result['render_time']:.2f}s)")
            class_paths.append(result["output_file"])
            total_class_render_time += result["render_time"]
        else:
            print(f"Class {result['class_name']}: Failed")
    
    if not successful_renders:
        print("No classes were rendered successfully. Exiting.")
        total_time = time.time() - total_start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        return
    
    # Calculate average render time per class
    avg_render_time = total_class_render_time / successful_renders if successful_renders > 0 else 0
    print(f"\nRendering statistics:")
    print(f"  - Successful renders: {successful_renders}/{len(animation_classes)}")
    print(f"  - Total parallel render time: {render_time:.2f} seconds")
    print(f"  - Average render time per class: {avg_render_time:.2f} seconds")
    print(f"  - Total cumulative render time: {total_class_render_time:.2f} seconds")
    print(f"  - Parallel speedup factor: {total_class_render_time/render_time if render_time > 0 else 0:.2f}x")
    
    # Find all rendered class files with improved search
    print("\n=== Finding All Rendered Class Files ===")
    valid_class_paths = find_class_files.remote()
    
    if not valid_class_paths:
        print("No valid class files found. Check if any rendering was successful.")
        # List all contents of the volume for debugging
        output_contents = list_output_directory.remote()
        print("\n=== Output Directory Contents ===")
        for line in output_contents:
            print(line)
        
        total_time = time.time() - total_start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        return
    
    print(f"\nFound {len(valid_class_paths)} valid class renderings:")
    for path in valid_class_paths:
        print(f"  - {path}")
    
    # Sort the paths to ensure consistent ordering
    valid_class_paths.sort()
    
    # Concatenate all class videos
    print("\n=== Concatenating Class Videos ===")
    concat_start_time = time.time()
    output_filename = f"combined_animations.mp4"
    output_path = f"/root/shared/output/{output_filename}"
    
    concat_result = concat_videos.remote(valid_class_paths, output_path)
    concat_time = time.time() - concat_start_time
    
    if concat_result["success"]:
        print(f"Successfully concatenated {concat_result['videos_count']} videos into: {concat_result['output_file']}")
        print(f"Concatenation time: {concat_time:.2f} seconds")
        
        # List the output directory to confirm the file exists
        output_contents = list_output_directory.remote()
        print("\n=== Output Directory Contents ===")
        for line in output_contents:
            print(line)
            
        total_time = time.time() - total_start_time
        print(f"\nRendering complete! Final output is stored in the volume at: {output_path}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("Failed to concatenate videos. Details:")
        print(f"Command used: {concat_result['command']}")
        print(f"Error: {concat_result.get('stderr', 'No error details available')}")
        
        total_time = time.time() - total_start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds (process failed)")