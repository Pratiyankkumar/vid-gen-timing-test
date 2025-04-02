import modal
import os
import subprocess
import time
import shutil
import threading
import json
import matplotlib.pyplot as plt
import pandas as pd

# Pre-define GPU environment variables
gpu_env = {
    "CUDA_VISIBLE_DEVICES": "0",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "MANIMGL_ENABLE_GPU": "1",
    "MANIMGL_MULTIPROCESSING": "0",
    "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
}

# Pre-define CPU environment variables
cpu_env = {
    "MANIMGL_ENABLE_GPU": "0",
    "MANIMGL_MULTIPROCESSING": "1",
    "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu",
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
        "pandas",
        "matplotlib",  # Add matplotlib for visualization
    )
    .add_local_dir(".", remote_path="/root/local")
)

app = modal.App("manim-gpu-benchmark", image=image)

volume = modal.Volume.from_name("manim-outputs", create_if_missing=True)


def monitor_gpu(output_file, stop_event, interval=1.0):
    timestamp_start = time.time()
    gpu_stats = []

    print(f"Starting GPU monitoring, saving to {output_file}")

    while not stop_event.is_set():
        nvidia_smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )

        for line in nvidia_smi.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 9:
                relative_time = time.time() - timestamp_start
                stats = {
                    "timestamp": parts[0],
                    "relative_time": relative_time,
                    "gpu_index": parts[1],
                    "gpu_utilization": parts[2],
                    "memory_utilization": parts[3],
                    "memory_total": parts[4],
                    "memory_used": parts[5],
                    "memory_free": parts[6],
                    "temperature": parts[7],
                    "power_draw": parts[8],
                }
                gpu_stats.append(stats)
                print(
                    f"GPU stats: Util={stats['gpu_utilization']}%, Mem={stats['memory_used']}/{stats['memory_total']}MB, Temp={stats['temperature']}°C"
                )

        time.sleep(interval)

    with open(output_file, "w") as f:
        json.dump(gpu_stats, f, indent=2)

    print(f"GPU monitoring complete, saved {len(gpu_stats)} samples to {output_file}")


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
    nvidia_info = subprocess.check_output(["nvidia-smi"], text=True)
    print(f"NVIDIA-SMI Output:\n{nvidia_info}")
    subprocess.run(["rm -rf /root/output/gpu/*"], shell=True)
    output_path = "/root/output/gpu"
    os.makedirs(output_path, exist_ok=True)
    monitoring_output = os.path.join(output_path, f"{scene_name}_GPU_monitoring.json")
    stop_monitoring = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_gpu,
        args=(monitoring_output, stop_monitoring, 2),
    )

    print("Running on GPU...")
    os.environ["DISPLAY"] = ":1"
    display_process = subprocess.Popen(["Xvfb", ":1", "-screen", "0", "1920x1080x24"])
    time.sleep(2)
    monitor_thread.start()
    print("GPU monitoring started")
    start_time = time.time()
    cmd = "cd /root/local && "
    for key, value in gpu_env.items():
        cmd += f"{key}={value} "

    # Add command body
    cmd += f"manimgl binary_search.py {scene_name} -o"

    print(f"Running command on GPU: {cmd}")

    # Execute command
    process = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, env=os.environ
    )
    print(f"Exit code: {process.returncode}")
    print(f"STDOUT: {process.stdout}")
    print(f"STDERR: {process.stderr}")
    elapsed_time = time.time() - start_time

    # 停止GPU监控
    stop_monitoring.set()
    monitor_thread.join()
    print("GPU monitoring stopped")

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
            "exit_code": process.returncode,
            "monitoring_file": monitoring_output
            if os.path.exists(monitoring_output)
            else None,
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
                dst_filename = (
                    f"{scene_name}_GPU.mp4"
                    if file.endswith(".mp4")
                    else f"{scene_name}_GPU.wav"
                )
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
            output_files.append(
                {"path": rel_path, "size": size_bytes, "full_path": file_path}
            )

    # 添加GPU监控文件到结果
    monitoring_rel_path = (
        os.path.relpath(monitoring_output, output_path)
        if os.path.exists(monitoring_output)
        else None
    )

    return {
        "device": "GPU",
        "elapsed_time": elapsed_time,
        "output_files": output_files,
        "exit_code": process.returncode,
        "monitoring_file": monitoring_rel_path,
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

    process = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, env=os.environ
    )
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
            "exit_code": process.returncode,
        }
    manim_output_dir = os.path.dirname(found_mp4_files[0])

    # Copy and rename output files
    output_files = []
    if os.path.exists(manim_output_dir):
        print(f"Files in output directory: {os.listdir(manim_output_dir)}")
        for file in os.listdir(manim_output_dir):
            if file.endswith(".mp4") or file.endswith(".wav"):
                src_path = os.path.join(manim_output_dir, file)
                dst_filename = (
                    f"{scene_name}_CPU.mp4"
                    if file.endswith(".mp4")
                    else f"{scene_name}_CPU.wav"
                )
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
            output_files.append(
                {"path": rel_path, "size": size_bytes, "full_path": file_path}
            )

    return {
        "device": "CPU",
        "elapsed_time": elapsed_time,
        "output_files": output_files,
        "exit_code": process.returncode,
    }


# 修改后的函数：生成GPU内存使用的报告和可视化
@app.function(
    volumes={"/root/output": volume},
)
def generate_gpu_monitoring_report(monitoring_file_path):
    """生成GPU监控报告，专注于显存占用情况可视化"""
    if not os.path.exists(monitoring_file_path):
        print(f"Monitoring file not found: {monitoring_file_path}")
        return None

    try:
        with open(monitoring_file_path, "r") as f:
            monitoring_data = json.load(f)

        if not monitoring_data:
            print("No GPU monitoring data found")
            return None

        # 计算平均值和最大值
        gpu_util_values = [float(entry["gpu_utilization"]) for entry in monitoring_data]
        mem_util_values = [float(entry["memory_utilization"]) for entry in monitoring_data]
        mem_used_values = [float(entry["memory_used"]) for entry in monitoring_data]
        mem_total_values = [float(entry["memory_total"]) for entry in monitoring_data]
        mem_free_values = [float(entry["memory_free"]) for entry in monitoring_data]
        relative_times = [float(entry["relative_time"]) for entry in monitoring_data]

        report = {
            "samples_count": len(monitoring_data),
            "duration_seconds": monitoring_data[-1]["relative_time"] - monitoring_data[0]["relative_time"]
            if len(monitoring_data) > 1
            else 0,
            "gpu_utilization": {
                "avg": sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else 0,
                "max": max(gpu_util_values) if gpu_util_values else 0,
                "min": min(gpu_util_values) if gpu_util_values else 0,
            },
            "memory_utilization": {
                "avg": sum(mem_util_values) / len(mem_util_values) if mem_util_values else 0,
                "max": max(mem_util_values) if mem_util_values else 0,
                "min": min(mem_util_values) if mem_util_values else 0,
            },
            "memory_used_mb": {
                "avg": sum(mem_used_values) / len(mem_used_values) if mem_used_values else 0,
                "max": max(mem_used_values) if mem_used_values else 0,
                "min": min(mem_used_values) if mem_used_values else 0,
            },
            "memory_total_mb": {
                "value": sum(mem_total_values) / len(mem_total_values) if mem_total_values else 0,
            },
            "memory_free_mb": {
                "avg": sum(mem_free_values) / len(mem_free_values) if mem_free_values else 0,
                "min": min(mem_free_values) if mem_free_values else 0,
            },
        }

        # 创建简单的CSV报告
        report_dir = os.path.dirname(monitoring_file_path)
        report_path = os.path.join(report_dir, "gpu_monitoring_report.csv")

        with open(report_path, "w") as f:
            f.write("Metric,Average,Maximum,Minimum\n")
            f.write(
                f"GPU Utilization (%),{report['gpu_utilization']['avg']:.2f},{report['gpu_utilization']['max']:.2f},{report['gpu_utilization']['min']:.2f}\n")
            f.write(
                f"Memory Utilization (%),{report['memory_utilization']['avg']:.2f},{report['memory_utilization']['max']:.2f},{report['memory_utilization']['min']:.2f}\n")
            f.write(
                f"Memory Used (MB),{report['memory_used_mb']['avg']:.2f},{report['memory_used_mb']['max']:.2f},{report['memory_used_mb']['min']:.2f}\n")
            f.write(
                f"Memory Free (MB),{report['memory_free_mb']['avg']:.2f},N/A,{report['memory_free_mb']['min']:.2f}\n")
            f.write(f"Memory Total (MB),{report['memory_total_mb']['value']:.2f},N/A,N/A\n")
            f.write(f"\nSamples Count,{report['samples_count']}\n")
            f.write(f"Duration (seconds),{report['duration_seconds']:.2f}\n")

        # 创建内存使用可视化图表
        # 创建DataFrame进行数据处理和可视化
        df = pd.DataFrame({
            'Time (s)': relative_times,
            'Memory Used (MB)': mem_used_values,
            'Memory Free (MB)': mem_free_values,
            'Memory Utilization (%)': mem_util_values,
            'GPU Utilization (%)': gpu_util_values
        })

        # 绘制内存使用随时间变化的图表
        plt.figure(figsize=(12, 10))

        # 1. 内存使用图 (显存占用MB)
        plt.subplot(2, 1, 1)
        plt.plot(df['Time (s)'], df['Memory Used (MB)'], 'b-', linewidth=2, label='Memory Used (MB)')
        plt.plot(df['Time (s)'], df['Memory Free (MB)'], 'g-', linewidth=2, label='Memory Free (MB)')
        plt.axhline(y=float(mem_total_values[0]), color='r', linestyle='--', linewidth=1,
                    label=f'Total Memory: {mem_total_values[0]} MB')
        plt.fill_between(df['Time (s)'], df['Memory Used (MB)'], alpha=0.3, color='blue')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory (MB)')
        plt.title('GPU Memory Usage Over Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 2. 内存和GPU利用率图
        plt.subplot(2, 1, 2)
        plt.plot(df['Time (s)'], df['Memory Utilization (%)'], 'b-', linewidth=2, label='Memory Utilization (%)')
        plt.plot(df['Time (s)'], df['GPU Utilization (%)'], 'r-', linewidth=2, label='GPU Utilization (%)')
        plt.fill_between(df['Time (s)'], df['Memory Utilization (%)'], alpha=0.3, color='blue')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Utilization (%)')
        plt.title('GPU and Memory Utilization Over Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()

        # 保存图表
        chart_path = os.path.join(report_dir, "gpu_memory_usage_chart.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"GPU monitoring report generated: {report_path}")
        print(f"GPU memory usage chart generated: {chart_path}")

        return report

    except Exception as e:
        print(f"Error generating GPU monitoring report: {e}")
        import traceback
        traceback.print_exc()
        return None


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
    for file_info in gpu_result["output_files"]:
        print(f"- {file_info['path']} ({file_info['size'] / 1024 / 1024:.2f} MB)")

    # 处理GPU监控数据
    if "monitoring_file" in gpu_result and gpu_result["monitoring_file"]:
        print("\n=== Generating GPU Monitoring Report ===")
        monitoring_file_path = os.path.join(f"/root/output/gpu", gpu_result["monitoring_file"])
        monitoring_report = generate_gpu_monitoring_report.remote(monitoring_file_path)

        if monitoring_report:
            print("\n=== GPU Memory Usage Stats ===")
            print(f"Total Memory: {monitoring_report['memory_total_mb']['value']:.2f} MB")
            print(
                f"Average Memory Usage: {monitoring_report['memory_used_mb']['avg']:.2f} MB ({monitoring_report['memory_utilization']['avg']:.2f}%)")
            print(
                f"Maximum Memory Usage: {monitoring_report['memory_used_mb']['max']:.2f} MB ({monitoring_report['memory_utilization']['max']:.2f}%)")
            print(f"Minimum Free Memory: {monitoring_report['memory_free_mb']['min']:.2f} MB")
            print(f"Average GPU Utilization: {monitoring_report['gpu_utilization']['avg']:.2f}%")

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
        for device, result in [("GPU", gpu_result), ("CPU", cpu_result)]:
            device_dir = os.path.join(local_output_dir, device.lower())
            os.makedirs(device_dir, exist_ok=True)

            for file_info in result["output_files"]:
                if file_info["path"].endswith(".mp4") or file_info["path"].endswith(".wav"):
                    print(f"Downloading {device} file: {file_info['path']}...")
                    content = download_file.remote(file_info["full_path"])
                    if content is not None:
                        local_file_path = os.path.join(device_dir, file_info["path"])
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                        with open(local_file_path, "wb") as f:
                            f.write(content)
                        print(f"✅ Downloaded to {local_file_path}")
                    else:
                        print(f"❌ Failed to download {file_info['path']}")

            if device == "GPU" and "monitoring_file" in result and result["monitoring_file"]:
                print(f"Downloading GPU monitoring data...")

                monitoring_path = os.path.join(f"/root/output/{device.lower()}", result["monitoring_file"])
                monitoring_content = download_file.remote(monitoring_path)
                if monitoring_content is not None:
                    local_monitoring_path = os.path.join(device_dir, result["monitoring_file"])
                    with open(local_monitoring_path, "wb") as f:
                        f.write(monitoring_content)
                    print(f"✅ Downloaded monitoring data to {local_monitoring_path}")

                # 下载GPU内存使用报告和图表
                report_path = os.path.join(f"/root/output/{device.lower()}", "gpu_monitoring_report.csv")
                report_content = download_file.remote(report_path)
                if report_content is not None:
                    local_report_path = os.path.join(device_dir, "gpu_monitoring_report.csv")
                    with open(local_report_path, "wb") as f:
                        f.write(report_content)
                    print(f"✅ Downloaded monitoring report to {local_report_path}")

                # 下载GPU内存使用图表
                chart_path = os.path.join(f"/root/output/{device.lower()}", "gpu_memory_usage_chart.png")
                chart_content = download_file.remote(chart_path)
                if chart_content is not None:
                    local_chart_path = os.path.join(device_dir, "gpu_memory_usage_chart.png")
                    with open(local_chart_path, "wb") as f:
                        f.write(chart_content)
                    print(f"✅ Downloaded memory usage chart to {local_chart_path}")

        print(f"\nAll files downloaded to {local_output_dir}")
        print("\n=== FINAL PERFORMANCE REPORT ===")
        print(f"GPU rendering time: {gpu_render_time:.2f} seconds")
        print(f"CPU rendering time: {cpu_render_time:.2f} seconds")
        print(f"Performance gain: GPU is {speedup:.2f}x faster than CPU")
        if "monitoring_file" in gpu_result and gpu_result["monitoring_file"]:
            print("\n=== GPU MEMORY USAGE SUMMARY ===")
            monitoring_report = generate_gpu_monitoring_report.remote(
                os.path.join(f"/root/output/gpu", gpu_result["monitoring_file"])
            )
            if monitoring_report:
                print(f"Samples collected: {monitoring_report['samples_count']}")
                print(f"Total Memory: {monitoring_report['memory_total_mb']['value']:.2f} MB")
                print(
                    f"Memory Usage: Avg={monitoring_report['memory_used_mb']['avg']:.2f}MB ({monitoring_report['memory_utilization']['avg']:.2f}%), Max={monitoring_report['memory_used_mb']['max']:.2f}MB ({monitoring_report['memory_utilization']['max']:.2f}%)")
                print(f"Memory Free: Min={monitoring_report['memory_free_mb']['min']:.2f}MB")
                print(
                    f"GPU Utilization: Avg={monitoring_report['gpu_utilization']['avg']:.2f}%, Max={monitoring_report['gpu_utilization']['max']:.2f}%")
                print("\nA visualization of the GPU memory usage has been generated as 'gpu_memory_usage_chart.png'")
    else:
        print("Cannot compare performance as at least one of the runs failed.")