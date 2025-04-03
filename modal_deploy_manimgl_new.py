import modal
import os
import subprocess
import time
import shutil

# GPU environment variables
gpu_env = {
    "PYOPENGL_PLATFORM": "egl",
    "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    "__NV_PRIME_RENDER_OFFLOAD": "1",
    "LIBGL_ALWAYS_SOFTWARE": "1",
    "__GL_SYNC_TO_VBLANK": "0",
    "DISPLAY": ":1",
    "MODERNGL_WINDOW":"egl",
    "XDG_RUNTIME_DIR": "/tmp/runtime-root"
}

# Create container image
image = (
    modal.Image.from_registry("robopaas/cudagl:12.2.0-devel-ubuntu22.04", add_python="3.11")
    .run_commands(
        "export DEBIAN_FRONTEND=noninteractive",
        "ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime",
        "apt-get update && apt-get install -y tzdata",
        "dpkg-reconfigure --frontend noninteractive tzdata",
    )
    .apt_install(
        "build-essential",
        "clang",
        "pkg-config",
        "python3-dev",
        "ffmpeg",
        "sox",
        "libsox-fmt-all",
        "xvfb",
        "x11-utils",
        "mesa-utils",
        "libegl1-mesa-dev",
        "libgles2-mesa-dev",
        "libglvnd-dev",
        "libglfw3-dev",
        "freeglut3-dev",
        "libgl1-mesa-dev",
        "libgl1-mesa-glx",
        "nvidia-utils-525",
        "libnvidia-gl-525",
        "libcairo2-dev",
        "libpango1.0-dev",
        "texlive",
        "texlive-latex-extra",
        "texlive-fonts-extra",
        "texlive-latex-recommended",
        "texlive-science",
        "texlive-fonts-recommended",
    )
    .run_commands(
        "mkdir -p /etc/OpenCL/vendors && echo \"libnvidia-opencl.so.1\" > /etc/OpenCL/vendors/nvidia.icd",
        "pip install --upgrade pip setuptools wheel build"
    )
    .pip_install(
        "manimgl",
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
        "pandas",
        "matplotlib",
        "PyOpenGL",
        "PyOpenGL-accelerate",
    )
    .add_local_dir(".", remote_path="/root/local")
)

app = modal.App("manimgl-gpu", image=image)
volume = modal.Volume.from_name("manim-opengl", create_if_missing=True)


@app.function(
    gpu="A10G",
    volumes={"/root/output": volume},
    timeout=1800,
)
def render_manim_gpu(scene_name="SimpleAnimation"):
    os.environ["PYTHONPATH"] = "/root/local"
    print(subprocess.getoutput("eglinfo || true"))
    for key, value in gpu_env.items():
        os.environ[key] = value

    print("GPU Environment Diagnostics:")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
    print(f"PYOPENGL_PLATFORM: {os.environ.get('PYOPENGL_PLATFORM')}")

    output_path = "/root/output/gpu"
    os.makedirs(output_path, exist_ok=True)

    display_process = subprocess.Popen([
        "Xvfb", ":1",
        "-screen", "0", "1280x720x24",
        "+extension", "GLX",
        "+render",
        "-ac"
    ])

    time.sleep(2)
    print("Running on GPU...")

    subprocess.check_call(["xdpyinfo", "-display", ":1"], stdout=subprocess.DEVNULL)
    print("Virtual display running properly")

    start_time = time.time()
    cmd = "cd /root/local && "
    for key, value in gpu_env.items():
        cmd += f"{key}={value} "
    cmd = (
        "cd /root/local && "
        "PYOPENGL_PLATFORM=egl manimgl binary_search_manimgl.py "
        f"{scene_name} -o"
    )
    print(f"Running command on GPU: {cmd}")
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=os.environ)
    print(f"Exit code: {process.returncode}")
    print(f"STDOUT: {process.stdout}")
    print(f"STDERR: {process.stderr}")
    elapsed_time = time.time() - start_time

    display_process.terminate()

    found_mp4_files = []
    search_dirs = [
        "/root/local/videos",
    ]

    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith(".mp4"):
                        found_mp4_files.append(os.path.join(root, file))
                        print(f"Found MP4 file: {os.path.join(root, file)}")

    if not found_mp4_files:
        print("Warning: No MP4 files found!")
        return {
            "device": "GPU",
            "elapsed_time": elapsed_time,
            "output_files": [],
            "exit_code": process.returncode,
        }

    output_files = []

    for src_path in found_mp4_files:
        dst_filename = f"{scene_name}_GPU.mp4"
        dst_path = os.path.join(output_path, dst_filename)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        print(f"Copying {src_path} to {dst_path}")
        shutil.copy2(src_path, dst_path)
        print(f"Copied output file to {dst_path}")

        size_bytes = os.path.getsize(dst_path)
        output_files.append({"path": dst_filename, "size": size_bytes, "full_path": dst_path})

    return {
        "device": "GPU",
        "elapsed_time": elapsed_time,
        "output_files": output_files,
        "exit_code": process.returncode,
    }


@app.function(
    volumes={"/root/output": volume},
    timeout=1800,
    cpu=8,
    memory=16384,
)
def render_manim_cpu(scene_name="SimpleAnimation"):
    cpu_env = {
        "MANIMGL_MULTIPROCESSING": "1",
        "DISPLAY": ":1"
    }

    for key, value in cpu_env.items():
        os.environ[key] = value

    print("CPU Environment Diagnostics:")
    print(f"MANIMGL_MULTIPROCESSING: {os.environ.get('MANIMGL_MULTIPROCESSING')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")

    output_path = "/root/output/cpu"
    os.makedirs(output_path, exist_ok=True)

    print("Running on CPU...")
    display_process = subprocess.Popen([
        "Xvfb", ":1",
        "-screen", "0", "1280x720x24",
        "+extension", "GLX",
        "+render",
        "-ac"
    ])
    time.sleep(2)

    print("✅ [Patch] Forcing standalone EGL context")
    start_time = time.time()
    cmd = "cd /root/local && "
    for key, value in gpu_env.items():
        cmd += f"{key}={value} "
    cmd += f"PYTHONPATH=/root/local python3 -c 'import patch_manim_camera' && manimgl binary_search_manimgl.py {scene_name} -o"
    print(f"Running command on CPU: {cmd}")
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=os.environ)
    print(f"Exit code: {process.returncode}")
    elapsed_time = time.time() - start_time

    display_process.terminate()

    found_mp4_files = []
    search_dirs = [
        "/root/local/videos",
    ]
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
            "exit_code": process.returncode,
        }

    output_files = []

    for src_path in found_mp4_files:
        dst_filename = f"{scene_name}_CPU.mp4"
        dst_path = os.path.join(output_path, dst_filename)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        print(f"Copying {src_path} to {dst_path}")
        shutil.copy2(src_path, dst_path)
        print(f"Copied output file to {dst_path}")

        size_bytes = os.path.getsize(dst_path)
        output_files.append({"path": dst_filename, "size": size_bytes, "full_path": dst_path})

    return {
        "device": "CPU",
        "elapsed_time": elapsed_time,
        "output_files": output_files,
        "exit_code": process.returncode,
    }


@app.function(
    volumes={"/root/output": volume},
)
def download_file(file_path):
    print(f"Attempting to download {file_path}")

    if os.path.exists(file_path):
        print(f"File exists: size = {os.path.getsize(file_path)} bytes")
        with open(file_path, "rb") as f:
            content = f.read()
            print(f"Read {len(content)} bytes")
            return content

    # File not found, try to find an alternative file
    parent_dir = os.path.dirname(file_path)
    filename = os.path.basename(file_path)

    print(f"Error: File {file_path} not found")

    # Search in possible locations
    search_dirs = [
        "/root/output/gpu",
        "/root/output/cpu",
        "/root/local",
        "/root/local/videos",
        "/root/local/media/videos"
    ]

    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"Checking directory: {search_dir}")
            for item in os.listdir(search_dir):
                item_path = os.path.join(search_dir, item)
                if os.path.isfile(item_path) and item.endswith(".mp4"):
                    print(f"Found possible alternative file: {item_path}")
                    with open(item_path, "rb") as f:
                        content = f.read()
                        print(f"Using alternative file, read {len(content)} bytes")
                        return content

    # Return empty content if no file found
    print("No usable file found, returning empty content")
    return b""


@app.local_entrypoint()
def main():
    print("Starting GPU vs CPU Benchmark Test...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scene_name = "SimpleAnimation"
    local_output_dir = os.path.join(script_dir, "downloaded_outputs")

    if os.path.exists(local_output_dir):
        shutil.rmtree(local_output_dir)

    os.makedirs(local_output_dir, exist_ok=True)
    os.makedirs(os.path.join(local_output_dir, "gpu"), exist_ok=True)
    os.makedirs(os.path.join(local_output_dir, "cpu"), exist_ok=True)

    print("\n=== Running ManimGL on GPU ===")
    gpu_start_time = time.time()
    gpu_result = render_manim_gpu.remote(scene_name)
    gpu_total_time = time.time() - gpu_start_time
    print(f"GPU total execution time: {gpu_total_time:.2f} seconds")
    print(f"GPU rendering time: {gpu_result['elapsed_time']:.2f} seconds")
    print(f"Exit code: {gpu_result['exit_code']}")
    print(f"Generated {len(gpu_result['output_files'])} output files")
    for file_info in gpu_result["output_files"]:
        print(f"- {file_info['path']} ({file_info['size'] / 1024 / 1024:.2f} MB)")

    print("\n=== Running ManimGL on CPU ===")
    cpu_start_time = time.time()
    cpu_result = render_manim_cpu.remote(scene_name)
    cpu_total_time = time.time() - cpu_start_time
    print(f"CPU total execution time: {cpu_total_time:.2f} seconds")
    print(f"CPU rendering time: {cpu_result['elapsed_time']:.2f} seconds")
    print(f"Exit code: {cpu_result['exit_code']}")
    print(f"Generated {len(cpu_result['output_files'])} output files")
    for file_info in cpu_result["output_files"]:
        print(f"- {file_info['path']} ({file_info['size'] / 1024 / 1024:.2f} MB)")

    print("\n=== GPU vs CPU Performance Comparison ===")
    if gpu_result and cpu_result:
        gpu_render_time = gpu_result["elapsed_time"]
        cpu_render_time = cpu_result["elapsed_time"]
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

        # Download GPU files
        print("Downloading GPU files...")
        gpu_output_path = "/root/output/gpu"
        for file_info in gpu_result["output_files"]:
            file_path = os.path.join(gpu_output_path, file_info["path"])
            local_path = os.path.join(local_output_dir, "gpu", file_info["path"])

            print(f"Downloading {file_path}...")
            content = download_file.remote(file_path)

            if content:
                with open(local_path, "wb") as f:
                    f.write(content)
                print(f"Downloaded to {local_path}")
            else:
                print(f"Failed to download {file_path}")

        # Download CPU files
        print("Downloading CPU files...")
        cpu_output_path = "/root/output/cpu"
        for file_info in cpu_result["output_files"]:
            file_path = os.path.join(cpu_output_path, file_info["path"])
            local_path = os.path.join(local_output_dir, "cpu", file_info["path"])

            print(f"Downloading {file_path}...")
            content = download_file.remote(file_path)

            if content:
                with open(local_path, "wb") as f:
                    f.write(content)
                print(f"Downloaded to {local_path}")
            else:
                print(f"Failed to download {file_path}")

        print(f"\nAll files downloaded to {local_output_dir}")
        print("\n=== FINAL PERFORMANCE REPORT ===")
        print(f"GPU rendering time: {gpu_render_time:.2f} seconds")
        print(f"CPU rendering time: {cpu_render_time:.2f} seconds")
        print(f"Performance gain: GPU is {speedup:.2f}x faster than CPU")
    else:
        print("Cannot compare performance as at least one of the runs failed.")