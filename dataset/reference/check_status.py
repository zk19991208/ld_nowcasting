import os, sys

pid = 1007410
try:
    os.kill(pid, 0)
    print(f"Process {pid} is RUNNING")
except ProcessLookupError:
    print(f"Process {pid} is NOT running")
except PermissionError:
    print(f"Process {pid} exists but no permission to signal")

# Check log file size
log_path = "/liyang/works/nowcast_case/experiments/DiffCast_full/run.log"
if os.path.exists(log_path):
    size = os.path.getsize(log_path)
    print(f"Log file size: {size} bytes")
    mtime = os.path.getmtime(log_path)
    import datetime
    print(f"Log last modified: {datetime.datetime.fromtimestamp(mtime)}")

# Check for checkpoints
ckpt_dir = "/liyang/works/nowcast_case/experiments/DiffCast_full/lightning_logs"
if os.path.exists(ckpt_dir):
    for root, dirs, files in os.walk(ckpt_dir):
        for f in files:
            fp = os.path.join(root, f)
            print(f"Found: {fp}")

# Check GPU usage
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    print(f"\nGPU Status:\n{result.stdout}")
except Exception as e:
    print(f"nvidia-smi error: {e}")
