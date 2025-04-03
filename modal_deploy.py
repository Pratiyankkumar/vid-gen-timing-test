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
        "manim",  # Specify exact version
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

app = modal.App("manim-direct-parallel-rendering", image=image)
# Create a single shared volume for all files
volume = modal.Volume.from_name("manim-outputs-parallel", create_if_missing=True)

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

def generate_segment_script(original_script_path, scene_name, segment_config):
    """
    Generate a modified Manim script that will only render a specific segment of the animation.
    """
    start_frame = segment_config["start_frame"]
    end_frame = segment_config["end_frame"]
    segment_id = segment_config["segment_id"]
    
    # Read the original script
    with open(original_script_path, "r") as f:
        script_content = f.read()
    
    # Create a segment-specific script with more accurate frame tracking
    segment_script = f"""
# Modified script for segment {segment_id} (frames {start_frame}-{end_frame})
import sys
import math
from manim import *

# Original script content
{script_content}

# Create a segment-specific scene that inherits from the original
class Segment{segment_id}({scene_name}):
    def construct(self):
        # Store the original play method
        original_play = self.play
        
        # Initialize frame tracking
        self.current_frame = 0
        self.segment_start = {start_frame}
        self.segment_end = {end_frame}
        
        def modified_play(self, *animations, **kwargs):
            # Get the run time for this animation
            run_time = kwargs.get('run_time', 1)
            anim_frames = math.ceil(run_time * config.frame_rate)
            
            # Determine if this animation is in our segment
            if self.current_frame + anim_frames <= self.segment_start:
                # Skip this animation - it's before our segment
                self.current_frame += anim_frames
                return self
            
            if self.current_frame >= self.segment_end:
                # Skip this animation - it's after our segment
                return self
            
            # Animation is at least partially in our segment - play it
            result = original_play(*animations, **kwargs)
            self.current_frame += anim_frames
            return result
        
        # Replace the play method
        self.play = modified_play.__get__(self, type(self))
        
        # Run the original construct method with our modified play
        super().construct()
        
        # Restore the original play method
        self.play = original_play

if __name__ == "__main__":
    # Use standard resolution and frame rate
    config.pixel_height = 720
    config.pixel_width = 1280
    config.frame_rate = 30
    Segment{segment_id}().render()
"""

    # Write the modified script to a temporary file
    segment_script_path = f"/tmp/segment_{segment_id}.py"
    with open(segment_script_path, "w") as f:
        f.write(segment_script)
    
    return segment_script_path

def get_animation_metadata(script_path, scene_name):
    """Get metadata about the animation to determine how to split it"""
    # Check for operating system
    import platform
    is_windows = platform.system() == "Windows"
    
    display_process = None
    try:
        # Set up virtual display (only on Linux/Unix)
        if not is_windows:
            os.environ["DISPLAY"] = ":1"
            display_process = subprocess.Popen(["Xvfb", ":1", "-screen", "0", "1920x1080x24"])
            time.sleep(2)
        
        # Run manim with --dry_run flag to get animation info
        cmd = f"cd {os.path.dirname(script_path)} && manim --dry_run {os.path.basename(script_path)} {scene_name}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # Parse the output to get animation duration and frames
        output = result.stdout + result.stderr
        print(f"Dry run output: {output}")
        
        # Default values if parsing fails
        total_frames = 300  # About 10 seconds at 30fps
        total_duration = 10
        frame_rate = 30
        
        # Try to parse the output
        for line in output.split('\n'):
            if "animations" in line.lower() and "frames" in line.lower():
                # Example: "Total: 10 animations, 300 frames"
                parts = line.split(',')
                if len(parts) >= 2:
                    frames_part = parts[1].strip()
                    frames_str = ''.join(filter(str.isdigit, frames_part))
                    if frames_str:
                        total_frames = int(frames_str)
                        break
            
        # Calculate total duration assuming 30fps
        total_duration = total_frames / frame_rate
        
        return {
            "total_frames": total_frames,
            "total_duration": total_duration,
            "frame_rate": frame_rate
        }
    finally:
        if display_process:
            display_process.terminate()

# Function to upload a script to the Modal environment
@app.function(volumes={"/root/shared": volume})
def upload_script(script_content, script_name):
    """Upload a script to the shared volume"""
    script_path = f"/root/shared/{script_name}"
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    print(f"Script uploaded to: {script_path}")
    return script_path

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

@app.function(
    volumes={"/root/shared": volume},
    timeout=1200,
    gpu="any" if os.environ.get("USE_GPU", "0") == "1" else None,
    cpu=4 if os.environ.get("USE_GPU", "0") == "0" else 2,
    memory=8192 if os.environ.get("USE_GPU", "0") == "0" else 4096,
)
def render_animation_segment(script_path, scene_name, segment_config):
    """Render a specific segment of the animation directly using Manim"""
    segment_id = segment_config["segment_id"]
    start_frame = segment_config["start_frame"]
    end_frame = segment_config["end_frame"]
    
    # Set up environment variables based on hardware
    has_gpu = os.environ.get("MODAL_CONTAINER_GPU_COUNT", "0") != "0"
    env_vars = gpu_env if has_gpu else cpu_env
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Configure output paths
    output_dir = f"/root/shared/output/segment_{segment_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up virtual display
    os.environ["DISPLAY"] = ":1"
    display_process = subprocess.Popen(["Xvfb", ":1", "-screen", "0", "1920x1080x24"])
    time.sleep(2)
    
    try:
        # Generate a modified script for this segment
        segment_script_path = generate_segment_script(script_path, scene_name, segment_config)
        print(f"Generated segment script at: {segment_script_path}")
        
        # Render the segment
        start_time = time.time()
        output_path = f"{output_dir}/segment_{segment_id}.mp4"
        
        # Run the script directly with Python
        cmd = f"python {segment_script_path}"
        print(f"Running command: {cmd}")
        
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Stdout: {process.stdout}")
        print(f"Stderr: {process.stderr}")
        
        render_time = time.time() - start_time
        
        # Find the rendered output file
        found_mp4_files = []
        search_dirs = ["/tmp", "./media", os.path.dirname(segment_script_path)]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith(".mp4") and f"Segment{segment_id}" in file:
                            found_mp4_files.append(os.path.join(root, file))
        
        if found_mp4_files:
            # Copy to output location
            segment_output_path = os.path.join(output_dir, f"segment_{segment_id}.mp4")
            shutil.copy2(found_mp4_files[0], segment_output_path)
            print(f"Copied segment {segment_id} to {segment_output_path}")
            
            return {
                "segment_id": segment_id,
                "success": True,
                "render_time": render_time,
                "output_file": segment_output_path,
            }
        else:
            print(f"No output file found for segment {segment_id}")
            return {
                "segment_id": segment_id,
                "success": False,
                "render_time": render_time,
                "output_file": None,
                "error": "No output file found"
            }
    finally:
        # Clean up the virtual display
        if display_process:
            display_process.terminate()


@app.function(
    volumes={"/root/shared": volume},
    timeout=600,
    cpu=4,
    memory=8192,
)
def combine_segments(segment_paths, output_filename):
    """Combine all rendered segments into a final video using ffmpeg with appropriate settings"""
    print(f"Combining segments from paths: {segment_paths}")
    
    if not segment_paths:
        print("Error: No segment paths provided!")
        return {
            "success": False,
            "output_file": None,
            "reason": "No segment paths provided"
        }
    
    # Sort the segment paths by segment ID to ensure correct order
    segment_paths.sort(key=lambda x: int(x.split('segment_')[1].split('/')[0]))
    
    # Create a temporary directory for intermediate files
    temp_dir = "/tmp/merge_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a file list for ffmpeg
    concat_file = f"{temp_dir}/concat_list.txt"
    with open(concat_file, "w") as f:
        for path in segment_paths:
            if os.path.exists(path):
                f.write(f"file '{path}'\n")
    
    # Output path for combined video
    output_path = f"/root/shared/output/{output_filename}"
    
    # Verify concat file contents
    with open(concat_file, "r") as f:
        print(f"Concat file contents:\n{f.read()}")
    
    # Use ffmpeg to concatenate videos with precise frame handling
    # cmd = f"ffmpeg -f concat -safe 0 -i {concat_file} -c copy {output_path}"
    # Use re-encoding to ensure frame consistency
    cmd = f"ffmpeg -f concat -safe 0 -i {concat_file} -c:v libx264 -preset medium -crf 22 {output_path}"
    
    print(f"Running command: {cmd}")
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    success = process.returncode == 0
    if success:
        print(f"Successfully combined segments into: {output_path}")
    else:
        print(f"Failed to combine segments: {process.stderr}")
    
    return {
        "success": success,
        "output_file": output_path if success else None,
        "segments_count": len(segment_paths),
        "stdout": process.stdout,
        "stderr": process.stderr
    }

@app.function(
    volumes={"/root/shared": volume},
)
def download_file(file_path):
    """Download a file from the volume to the local machine"""
    print(f"Attempting to download {file_path}")
    
    if os.path.exists(file_path):
        print(f"File size: {os.path.getsize(file_path)} bytes")
        with open(file_path, "rb") as f:
            content = f.read()
            print(f"Read {len(content)} bytes")
            return content
    else:
        print(f"ERROR: File {file_path} does not exist")
        parent_dir = os.path.dirname(file_path)
        if os.path.exists(parent_dir):
            contents = list_directory_contents(parent_dir)
            print(f"Contents of {parent_dir}:")
            for line in contents:
                print(line)
        else:
            print(f"Parent directory {parent_dir} does not exist")
        return None

# MOVED FUNCTIONS TO GLOBAL SCOPE
@app.function(volumes={"/root/shared": volume})
def find_segment_files():
    valid_segment_paths = []
    volume_base = "/root/shared/output"
    
    # List all segment directories
    if os.path.exists(volume_base):
        print(f"Scanning directories in {volume_base}:")
        for segment_dir in sorted(os.listdir(volume_base)):
            if segment_dir.startswith("segment_"):
                segment_path = os.path.join(volume_base, segment_dir)
                print(f"Checking directory: {segment_path}")
                
                # Find all MP4 files in this directory and its subdirectories
                for root, dirs, files in os.walk(segment_path):
                    for file in files:
                        if file.endswith(".mp4"):
                            full_path = os.path.join(root, file)
                            print(f"Found MP4: {full_path} ({os.path.getsize(full_path)} bytes)")
                            valid_segment_paths.append(full_path)
    else:
        print(f"Volume base directory {volume_base} does not exist")
    
    return valid_segment_paths

@app.function(volumes={"/root/shared": volume}, timeout=600)
def combine_segments_improved(segment_paths, output_path):
    """Combine all rendered segments into a final video using ffmpeg"""
    # Create a temporary directory for intermediate files
    temp_dir = "/tmp/merge_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a file list for ffmpeg
    concat_file = f"{temp_dir}/concat_list.txt"
    with open(concat_file, "w") as f:
        for path in segment_paths:
            if os.path.exists(path):
                f.write(f"file '{path}'\n")
    
    # Verify concat file contents
    with open(concat_file, "r") as f:
        print(f"Concat file contents:\n{f.read()}")
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use ffmpeg to concatenate videos with precise frame handling
    cmd = f"ffmpeg -f concat -safe 0 -i {concat_file} -c:v libx264 -preset medium -crf 22 {output_path}"
    
    print(f"Running command: {cmd}")
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    success = process.returncode == 0
    
    if success:
        print(f"Successfully combined segments into: {output_path}")
        print(f"Output file size: {os.path.getsize(output_path)} bytes")
    else:
        print(f"Failed to combine segments: {process.stderr}")
    
    # Return result
    return {
        "success": success,
        "output_file": output_path if success else None,
        "segments_count": len(segment_paths),
        "stdout": process.stdout,
        "stderr": process.stderr,
        "command": cmd
    }

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
    scene_name="SimpleAnimation", 
    num_segments=6,
    use_gpu=False
):
    """Main entry point for direct parallel rendering"""
    
    num_segments = int(num_segments)
    
    if use_gpu:
        os.environ["USE_GPU"] = "1"
    
    print(f"Starting Direct Parallel Rendering with {'GPU' if use_gpu else 'CPU'}...")
    
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
    
    # Get animation metadata
    print("\n=== Analyzing Animation ===")
    animation_info = get_animation_metadata(local_script_path, scene_name)
    total_frames = animation_info["total_frames"]
    
    # Add buffer to total frames to ensure we capture everything
    total_frames = int(total_frames * 1.1)  # Add 10% buffer
    print(f"Total frames (with buffer): {total_frames}")
    
    # Create segments with overlap between them
    frames_per_segment = total_frames // num_segments
    overlap = int(frames_per_segment * 0.1)  # 10% overlap
    
    segments = []
    for i in range(num_segments):
        # Calculate segment boundaries with overlap
        start_frame = max(0, i * frames_per_segment - overlap)
        end_frame = min(total_frames, (i + 1) * frames_per_segment + overlap)
        
        segments.append({
            "segment_id": i,
            "start_frame": start_frame,
            "end_frame": end_frame,
        })
        print(f"Segment {i}: frames {start_frame} to {end_frame}")
    
    # Render segments in parallel
    print("\n=== Rendering Segments in Parallel ===")
    start_time = time.time()
    segment_results = render_animation_segment.map(
        [verified_script_path] * num_segments, 
        [scene_name] * num_segments, 
        segments
    )
    render_time = time.time() - start_time
    
    # Process results
    segment_paths = []
    for result in segment_results:
        if result["success"]:
            print(f"Segment {result['segment_id']}: Success ({result['render_time']:.2f}s)")
            segment_paths.append(result["output_file"])
        else:
            print(f"Segment {result['segment_id']}: Failed")
    
    if not segment_paths:
        print("No segments were rendered successfully. Exiting.")
        return
    
    # Search for the MP4 files in each segment directory
    print("\n=== Finding Segment Files ===")
    valid_segment_paths = find_segment_files.remote()
    
    if not valid_segment_paths:
        print("No valid segment files found. Exiting.")
        return
    
    # Sort by segment number, ensuring correct order for concatenation
    valid_segment_paths.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x).split('_')[0]))) if ''.join(filter(str.isdigit, os.path.basename(x).split('_')[0])) else 0)
    
    print(f"\nFound {len(valid_segment_paths)} valid segments:")
    for path in valid_segment_paths:
        print(f"  - {path}")
    
    # Combine segments directly into volume
    print("\n=== Combining Segments ===")
    output_filename = f"{scene_name}_combined.mp4"
    output_path = f"/root/shared/output/{output_filename}"
    
    # Combine the segments
    combine_result = combine_segments_improved.remote(valid_segment_paths, output_path)
    
    if combine_result["success"]:
        print(f"Successfully combined segments into: {combine_result['output_file']}")
        
        # List the output directory to confirm the file exists
        output_contents = list_output_directory.remote()
        print("\n=== Output Directory Contents ===")
        for line in output_contents:
            print(line)
            
        print(f"\nRendering complete! Final output is stored in the volume at: {output_path}")
    else:
        print("Failed to combine segments. Details:")
        print(f"Command used: {combine_result['command']}")
        print(f"Error: {combine_result.get('stderr', 'No error details available')}")