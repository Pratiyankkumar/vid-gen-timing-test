import modal
import os
import subprocess
import time
import shutil

gpu_env = {
    "PYOPENGL_PLATFORM": "glx",
    "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    "__NV_PRIME_RENDER_OFFLOAD": "1",
    "__GL_SYNC_TO_VBLANK": "0",
    "DISPLAY": ":1",
    "MODERNGL_WINDOW":"glx",
}

init_commands = [
    "export DEBIAN_FRONTEND=noninteractive",
    "echo 'keyboard-configuration keyboard-configuration/layout select English (US)' | debconf-set-selections",
    "echo 'keyboard-configuration keyboard-configuration/variant select English (US)' | debconf-set-selections",
    "ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime",
    "apt-get update && apt-get install -y tzdata",
    "dpkg-reconfigure --frontend noninteractive tzdata",
    "apt-get update && apt-get install -y software-properties-common wget apt-utils",
    "add-apt-repository -y ppa:graphics-drivers/ppa",
    "apt-get update",
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.10")
    .run_commands(*init_commands)
    .apt_install(
        "build-essential",
        "clang",
        "pkg-config",
        "python3-dev",
        "xserver-xorg-video-dummy",
        "xvfb",
        "ffmpeg",
        "sox",
        "libsox-fmt-all",
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
        "libxrender1",
        "libxext6",
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

app = modal.App("manim-opengl", image=image)
volume = modal.Volume.from_name("manim-opengl", create_if_missing=True)

def create_xorg_conf():
    with open('/etc/X11/xorg.conf', 'w') as f:
        f.write('Section "ServerLayout"\n')
        f.write('    Identifier "Layout0"\n')
        f.write('    Screen 0 "Screen0" 0 0\n')
        f.write('EndSection\n\n')

        f.write('Section "Device"\n')
        f.write('    Identifier "Device0"\n')
        f.write('    Driver "nvidia"\n')
        f.write('    Option "AllowEmptyInitialConfiguration" "True"\n')
        f.write('EndSection\n\n')

        f.write('Section "Screen"\n')
        f.write('    Identifier "Screen0"\n')
        f.write('    Device "Device0"\n')
        f.write('    DefaultDepth 24\n')
        f.write('    Option "UseDisplayDevice" "None"\n')
        f.write('    SubSection "Display"\n')
        f.write('        Depth 24\n')
        f.write('        Virtual 1920 1080\n')
        f.write('    EndSubSection\n')
        f.write('EndSection\n')

@app.function(
    gpu="A10G",
    volumes={"/root/output": volume},
    timeout=1800,
)
def render_manim_gpu(scene_name="SimpleAnimation"):
    for key, value in gpu_env.items():
        os.environ[key] = value

    output_path = "/root/output/gpu"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    create_xorg_conf()
    subprocess.run(["chmod", "777", "/etc/X11/xorg.conf"])
    import threading
    stop_monitoring = threading.Event()

    def monitor_gpu():
        while not stop_monitoring.is_set():
            subprocess.run(["nvidia-smi"])
            time.sleep(5)

    monitor_thread = threading.Thread(target=monitor_gpu)
    monitor_thread.daemon = True
    monitor_thread.start()

    display_process = subprocess.Popen([
        "Xorg", "-noreset", "+extension", "GLX",
        "-logfile", "/tmp/xorg.log", "-config", "/etc/X11/xorg.conf",
        ":1"
    ])
    time.sleep(5)

    print("Running on GPU...")
    subprocess.run(["xdpyinfo", "-display", ":1"])
    print("Display :1 is available")

    start_time = time.time()

    cmd = "cd /root/local && "
    for key, value in gpu_env.items():
        cmd += f"{key}={value} "

    direct_cmd = cmd + f"manimgl binary_search_manimgl.py {scene_name} -o"
    print(f"Running command on GPU: {direct_cmd}")
    process = subprocess.run(direct_cmd, shell=True, capture_output=True, text=True, env=os.environ)
    elapsed_time = time.time() - start_time
    stop_monitoring.set()
    monitor_thread.join(timeout=1)

    display_process.terminate()
    subprocess.run("sync", shell=True)

    found_mp4_files = []
    search_dirs = [
        "/root/local/videos",
        "/root/output/gpu"
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
            "stdout": process.stdout,
            "stderr": process.stderr
        }

    output_files = []

    dst_filename = f"{scene_name}_GPU.mp4"
    dst_path = os.path.join(output_path, dst_filename)

    if os.path.exists(dst_path):
        os.remove(dst_path)

    if found_mp4_files:
        src_path = found_mp4_files[0]
        print(f"Copying {src_path} to {dst_path}")
        shutil.copy2(src_path, dst_path)
        subprocess.run("sync", shell=True)

        size_bytes = os.path.getsize(dst_path)
        output_files.append({
            "path": dst_filename,
            "size": size_bytes,
            "full_path": dst_path
        })

    return {
        "device": "GPU",
        "elapsed_time": elapsed_time,
        "output_files": output_files,
        "exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr
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
        "DISPLAY": ":1",
        "PYOPENGL_PLATFORM": "glx",
    }

    for key, value in cpu_env.items():
        os.environ[key] = value

    output_path = "/root/output/cpu"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    create_xorg_conf()

    print("Running on CPU...")

    display_process = subprocess.Popen([
        "Xorg", "-noreset",
        "-config", "/etc/X11/xorg.conf",
        "-logfile", "/tmp/xorg_cpu.log",
        ":1"
    ])
    time.sleep(5)

    subprocess.run(["xdpyinfo", "-display", ":1"])
    print("Display :1 is available")

    start_time = time.time()

    cmd = "cd /root/local && "
    for key, value in cpu_env.items():
        cmd += f"{key}={value} "
    cmd += f"manimgl binary_search_manimgl.py {scene_name} -o"

    print(f"Running command on CPU: {cmd}")
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=os.environ)
    elapsed_time = time.time() - start_time

    display_process.terminate()
    subprocess.run("sync", shell=True)

    found_mp4_files = []
    search_dirs = [
        "/root/local/videos",
        "/root/output/cpu"
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
            "stdout": process.stdout,
            "stderr": process.stderr
        }

    output_files = []

    dst_filename = f"{scene_name}_CPU.mp4"
    dst_path = os.path.join(output_path, dst_filename)

    if os.path.exists(dst_path):
        os.remove(dst_path)

    if found_mp4_files:
        src_path = found_mp4_files[0]
        print(f"Copying {src_path} to {dst_path}")
        shutil.copy2(src_path, dst_path)
        subprocess.run("sync", shell=True)

        size_bytes = os.path.getsize(dst_path)
        output_files.append({
            "path": dst_filename,
            "size": size_bytes,
            "full_path": dst_path
        })

    return {
        "device": "CPU",
        "elapsed_time": elapsed_time,
        "output_files": output_files,
        "exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr
    }


@app.function(
    volumes={"/root/output": volume},
)
def sync_volume():
    print("Syncing volume...")
    subprocess.run("sync", shell=True)
    return True


@app.function(
    volumes={"/root/output": volume},
)
def download_file(file_path):
    print(f"Downloading {file_path}")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            content = f.read()
            print(f"Read {len(content)} bytes")
            return content

    print(f"File {file_path} not found")
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
    if os.path.exists(os.path.join(local_output_dir, "gpu")):
        shutil.rmtree(os.path.join(local_output_dir, "gpu"))
    if os.path.exists(os.path.join(local_output_dir, "cpu")):
        shutil.rmtree(os.path.join(local_output_dir, "cpu"))

    videos_dir = "/root/local/videos"
    if os.path.exists(videos_dir):
        shutil.rmtree(videos_dir)

    os.makedirs(os.path.join(local_output_dir, "gpu"), exist_ok=True)
    os.makedirs(os.path.join(local_output_dir, "cpu"), exist_ok=True)

    print("\n=== Running ManimGL on GPU ===")
    gpu_start_time = time.time()
    gpu_result = render_manim_gpu.remote(scene_name)
    gpu_total_time = time.time() - gpu_start_time
    print(f"GPU total execution time: {gpu_total_time:.2f} seconds")
    print(f"GPU rendering time: {gpu_result['elapsed_time']:.2f} seconds")
    print("===============GPU Result===============")
    print(gpu_result)

    print(f"Generated {len(gpu_result['output_files'])} output files")
    for file_info in gpu_result["output_files"]:
        print(f"- {file_info['path']} ({file_info['size'] / 1024 / 1024:.2f} MB)")

    print("\n=== Running ManimGL on CPU ===")
    cpu_start_time = time.time()
    cpu_result = render_manim_cpu.remote(scene_name)
    cpu_total_time = time.time() - cpu_start_time
    print(f"CPU total execution time: {cpu_total_time:.2f} seconds")
    print(f"CPU rendering time: {cpu_result['elapsed_time']:.2f} seconds")
    print("===============CPU Result===============")
    print(cpu_result)
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

        print("\n=== Syncing files to volume ===")
        sync_volume.remote()

        print("\n=== Downloading files to local machine ===")

        print("Downloading GPU files...")
        for file_info in gpu_result["output_files"]:
            file_path = file_info["full_path"]
            local_path = os.path.join(local_output_dir, "gpu", file_info["path"])
            content = download_file.remote(file_path)
            if content:
                with open(local_path, "wb") as f:
                    f.write(content)
                print(f"Downloaded to {local_path}")
            else:
                print(f"Failed to download {file_path}")

        print("Downloading CPU files...")
        for file_info in cpu_result["output_files"]:
            file_path = file_info["full_path"]
            local_path = os.path.join(local_output_dir, "cpu", file_info["path"])
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