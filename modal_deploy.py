import modal
import os
import subprocess
import time
import shutil

# Pre-define GPU environment variables
gpu_env = {
    "CUDA_VISIBLE_DEVICES": "0",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "MANIMGL_ENABLE_GPU": "1",
    "MANIMGL_MULTIPROCESSING": "0",
    "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
}

# Pre-define CPU environment variables
cpu_env = {
    "MANIMGL_ENABLE_GPU": "0",
    "MANIMGL_MULTIPROCESSING": "1",
    "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu"
}

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .run_commands(
        "export DEBIAN_FRONTEND=noninteractive",
        "ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime",
        "apt-get update && apt-get install -y tzdata",
        "dpkg-reconfigure --frontend noninteractive tzdata",
    )
    .apt_install(
        "clang",
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
        "manimgl==1.7.2",
        "numpy",
        "manimpango",
        "requests",
        "moviepy",
        "torch",
        "pycairo",
        "pyglet",
        "pydub",
        "moderngl",
        "moderngl-window",
        "screeninfo",
        "mapbox-earcut",
        "validators",
        "tqdm",
    )
    .add_local_dir(
        ".",
        remote_path="/root/local"
    )
)

app = modal.App("manim-gpu-benchmark", image=image)

volume = modal.Volume.from_name("manim-outputs", create_if_missing=True)


@app.function(
    gpu="A10G",
    volumes={"/root/output": volume},
    timeout=1800,
)
def render_manim_gpu(scene_name="SimpleAnimation"):
    # Set environment variables
    for key, value in gpu_env.items():
        os.environ[key] = value

    print("GPU Environment Diagnostics:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"MANIMGL_ENABLE_GPU: {os.environ.get('MANIMGL_ENABLE_GPU')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")

    # Verify GPU availability
    nvidia_info = subprocess.check_output(["nvidia-smi"], text=True)
    print(f"NVIDIA-SMI Output:\n{nvidia_info}")

    # Clean output directory
    subprocess.run(["rm -rf /root/output/gpu/*"], shell=True)
    output_path = "/root/output/gpu"
    os.makedirs(output_path, exist_ok=True)

    print("Running on GPU...")

    # Setup virtual display
    os.environ["DISPLAY"] = ":1"
    display_process = subprocess.Popen(["Xvfb", ":1", "-screen", "0", "1920x1080x24"])
    time.sleep(2)

    # Start timing
    start_time = time.time()

    # Run command, ensuring all environment variables are passed
    cmd = "cd /root/local && "

    # Add all environment variables to command
    for key, value in gpu_env.items():
        cmd += f"{key}={value} "

    # Add command body
    cmd += f"manimgl binary_search.py {scene_name} -o"

    print(f"Running command on GPU: {cmd}")

    # Execute command
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=os.environ)
    print(f"Exit code: {process.returncode}")
    print(f"STDOUT: {process.stdout}")
    print(f"STDERR: {process.stderr}")
    elapsed_time = time.time() - start_time
    display_process.terminate()



    found_mp4_files = []
    search_dirs = ["/root/local", "/root/local/videos", "/root"]

    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith(".mp4"):
                        found_mp4_files.append(os.path.join(root, file))
                        print(f"Found MP4: {os.path.join(root, file)}")

    if not found_mp4_files:
        print("Warning: No MP4 files found!")
        return {
            "device": "GPU",
            "elapsed_time": elapsed_time,
            "output_files": [],
            "exit_code": process.returncode
        }

    # Get output directory
    manim_output_dir = os.path.dirname(found_mp4_files[0])

    # Copy and rename output files
    output_files = []
    if os.path.exists(manim_output_dir):
        print(f"Files in output directory: {os.listdir(manim_output_dir)}")
        for file in os.listdir(manim_output_dir):
            if file.endswith(".mp4") or file.endswith(".wav"):
                src_path = os.path.join(manim_output_dir, file)
                dst_filename = f"{scene_name}_GPU.mp4" if file.endswith(".mp4") else f"{scene_name}_GPU.wav"
                dst_path = os.path.join(output_path, dst_filename)
                print(f"Copying {src_path} to {dst_path}")
                shutil.copy2(src_path, dst_path)
                print(f"Copied output file to {dst_path}")

    # Get output file list
    for root, dirs, files in os.walk(output_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, output_path)
            size_bytes = os.path.getsize(file_path)
            output_files.append({
                "path": rel_path,
                "size": size_bytes,
                "full_path": file_path
            })

    return {
        "device": "GPU",
        "elapsed_time": elapsed_time,
        "output_files": output_files,
        "exit_code": process.returncode
    }


@app.function(
    volumes={"/root/output": volume},
    timeout=1800,
    cpu=8,
    memory=16384,
)
def render_manim_cpu(scene_name="SimpleAnimation"):
    # Set environment variables
    for key, value in cpu_env.items():
        os.environ[key] = value

    print("CPU Environment Diagnostics:")
    print(f"MANIMGL_ENABLE_GPU: {os.environ.get('MANIMGL_ENABLE_GPU')}")
    print(f"MANIMGL_MULTIPROCESSING: {os.environ.get('MANIMGL_MULTIPROCESSING')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    output_path = "/root/output/cpu"
    os.makedirs(output_path, exist_ok=True)
    print("Running on CPU...")
    os.environ["DISPLAY"] = ":1"
    display_process = subprocess.Popen(["Xvfb", ":1", "-screen", "0", "1920x1080x24"])
    time.sleep(2)
    subprocess.check_call(["xdpyinfo", "-display", ":1"], stdout=subprocess.DEVNULL)
    print("Virtual display running properly")
    start_time = time.time()
    cmd = "cd /root/local && "
    for key, value in cpu_env.items():
        cmd += f"{key}={value} "
    cmd += f"manimgl binary_search.py {scene_name} -o"
    print(f"Running command on CPU: {cmd}")

    process = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=os.environ)
    print(f"Exit code: {process.returncode}")
    print(f"STDOUT: {process.stdout}")
    print(f"STDERR: {process.stderr}")
    elapsed_time = time.time() - start_time
    display_process.terminate()


    found_mp4_files = []
    search_dirs = ["/root/local", "/root/local/videos", "/root"]

    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith(".mp4"):
                        found_mp4_files.append(os.path.join(root, file))
                        print(f"Found MP4: {os.path.join(root, file)}")

    if not found_mp4_files:
        print("Warning: No MP4 files found!")
        return {
            "device": "CPU",
            "elapsed_time": elapsed_time,
            "output_files": [],
            "exit_code": process.returncode
        }
    manim_output_dir = os.path.dirname(found_mp4_files[0])

    # Copy and rename output files
    output_files = []
    if os.path.exists(manim_output_dir):
        print(f"Files in output directory: {os.listdir(manim_output_dir)}")
        for file in os.listdir(manim_output_dir):
            if file.endswith(".mp4") or file.endswith(".wav"):
                src_path = os.path.join(manim_output_dir, file)
                dst_filename = f"{scene_name}_CPU.mp4" if file.endswith(".mp4") else f"{scene_name}_CPU.wav"
                dst_path = os.path.join(output_path, dst_filename)
                print(f"Copying {src_path} to {dst_path}")
                shutil.copy2(src_path, dst_path)
                print(f"Copied output file to {dst_path}")

    # Get output file list
    for root, dirs, files in os.walk(output_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, output_path)
            size_bytes = os.path.getsize(file_path)
            output_files.append({
                "path": rel_path,
                "size": size_bytes,
                "full_path": file_path
            })

    return {
        "device": "CPU",
        "elapsed_time": elapsed_time,
        "output_files": output_files,
        "exit_code": process.returncode
    }


@app.function(
    volumes={"/root/output": volume},
)
def download_file(file_path):
    print(f"Attempting to download {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
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
            print(f"Contents of {parent_dir}:")
            for item in os.listdir(parent_dir):
                full_path = os.path.join(parent_dir, item)
                size = os.path.getsize(full_path) if os.path.isfile(full_path) else "DIR"
                print(f"  - {item} ({size})")
        else:
            print(f"Parent directory {parent_dir} does not exist")
        return None


@app.local_entrypoint()
def main():
    print("Starting GPU vs CPU Benchmark Test...")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Use SimpleAnimation scene
    scene_name = "SimpleAnimation"

    local_output_dir = os.path.join(script_dir, "downloaded_outputs")
    os.makedirs(local_output_dir, exist_ok=True)

    print("\n=== Running ManimGL on GPU ===")

    gpu_start_time = time.time()
    gpu_result = render_manim_gpu.remote(scene_name)
    gpu_total_time = time.time() - gpu_start_time

    print(f"GPU total execution time: {gpu_total_time:.2f} seconds")
    print(f"GPU rendering time: {gpu_result['elapsed_time']:.2f} seconds")
    print(f"Exit code: {gpu_result['exit_code']}")
    print(f"Generated {len(gpu_result['output_files'])} output files")
    for file_info in gpu_result['output_files']:
        print(f"- {file_info['path']} ({file_info['size'] / 1024 / 1024:.2f} MB)")

    print("\n=== Running ManimGL on CPU ===")

    cpu_start_time = time.time()
    cpu_result = render_manim_cpu.remote(scene_name)
    cpu_total_time = time.time() - cpu_start_time

    print(f"CPU total execution time: {cpu_total_time:.2f} seconds")
    print(f"CPU rendering time: {cpu_result['elapsed_time']:.2f} seconds")
    print(f"Exit code: {cpu_result['exit_code']}")
    print(f"Generated {len(cpu_result['output_files'])} output files")

    for file_info in cpu_result['output_files']:
        print(f"- {file_info['path']} ({file_info['size'] / 1024 / 1024:.2f} MB)")

    print("\n=== GPU vs CPU Performance Comparison ===")
    if gpu_result and cpu_result:
        gpu_render_time = gpu_result['elapsed_time']
        cpu_render_time = cpu_result['elapsed_time']

        if gpu_render_time > 0:
            speedup = cpu_render_time / gpu_render_time
            print(f"GPU rendering time: {gpu_render_time:.2f} seconds")
            print(f"CPU rendering time: {cpu_render_time:.2f} seconds")
            print(f"Speedup: {speedup:.2f}x (GPU is {speedup:.2f} times faster than CPU)")

            print(f"GPU total execution time: {gpu_total_time:.2f} seconds")
            print(f"CPU total execution time: {cpu_total_time:.2f} seconds")
            total_speedup = cpu_total_time / gpu_total_time
            print(f"Total speedup: {total_speedup:.2f}x")
        else:
            print("Invalid GPU rendering time (0 or negative)")

        print("\n=== Downloading files to local machine ===")
        for device, result in [("GPU", gpu_result), ("CPU", cpu_result)]:
            device_dir = os.path.join(local_output_dir, device.lower())
            os.makedirs(device_dir, exist_ok=True)

            for file_info in result['output_files']:
                if file_info['path'].endswith('.mp4') or file_info['path'].endswith('.wav'):
                    print(f"Downloading {device} file: {file_info['path']}...")
                    content = download_file.remote(file_info['full_path'])
                    if content is not None:
                        local_file_path = os.path.join(device_dir, file_info['path'])
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                        with open(local_file_path, "wb") as f:
                            f.write(content)
                        print(f"✅ Downloaded to {local_file_path}")
                    else:
                        print(f"❌ Failed to download {file_info['path']}")

        print(f"\nAll files downloaded to {local_output_dir}")
        print("\n=== FINAL PERFORMANCE REPORT ===")
        print(f"GPU rendering time: {gpu_render_time:.2f} seconds")
        print(f"CPU rendering time: {cpu_render_time:.2f} seconds")
        print(f"Performance gain: GPU is {speedup:.2f}x faster than CPU")
    else:
        print("Cannot compare performance as at least one of the runs failed.")