import platform
import psutil
import subprocess
import shutil

def get_ram_info():
    """Returns total, available RAM and ZRAM in GB."""
    mem = psutil.virtual_memory()
    zram_gb = 0
    
    if platform.system() == "Linux":
        try:
            # Check for zram in /proc/swaps
            with open("/proc/swaps", "r") as f:
                lines = f.readlines()[1:] # Skip header
                for line in lines:
                    parts = line.split()
                    if "zram" in parts[0]:
                        zram_gb += float(parts[2]) / (1024**2) # Size is in KB
        except Exception:
            pass
            
    return {
        "total": mem.total / (1024**3),
        "available": mem.available / (1024**3),
        "zram": zram_gb
    }

def get_cpu_info():
    """Returns CPU model and thread count."""
    return {
        "model": platform.processor(),
        "threads": psutil.cpu_count(logical=True),
        "cores": psutil.cpu_count(logical=False)
    }

def get_gpu_info():
    """Detects NVIDIA or AMD GPUs."""
    gpus = []
    
    # Check NVIDIA
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            res = subprocess.check_output([nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], encoding="utf-8")
            for line in res.strip().split("\n"):
                name, vram = line.split(",")
                gpus.append({"vendor": "NVIDIA", "name": name.strip(), "vram_total": float(vram) / 1024})
        except Exception:
            pass

    # Check AMD (Linux)
    if platform.system() == "Linux":
        rocm_smi = shutil.which("rocm-smi")
        if rocm_smi:
            try:
                # Basic check, rocm-smi output varies
                gpus.append({"vendor": "AMD", "name": "ROCm-compatible GPU", "vram_total": 0})
            except Exception:
                pass
        
    return gpus

def detect_system():
    return {
        "os": platform.system(),
        "release": platform.release(),
        "ram": get_ram_info(),
        "cpu": get_cpu_info(),
        "gpus": get_gpu_info()
    }

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    
    sys_info = detect_system()
    console = Console()
    
    table = Table(title="Hardware Detection")
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("OS", f"{sys_info['os']} {sys_info['release']}")
    table.add_row("CPU", sys_info['cpu']['model'])
    table.add_row("Threads", str(sys_info['cpu']['threads']))
    table.add_row("RAM Total", f"{sys_info['ram']['total']:.2f} GB")
    table.add_row("RAM Available", f"{sys_info['ram']['available']:.2f} GB")
    table.add_row("Swap Total", f"{sys_info['ram']['swap_total']:.2f} GB")
    
    for gpu in sys_info['gpus']:
        table.add_row("GPU", f"{gpu['vendor']} {gpu['name']} ({gpu['vram_total']:.2f} GB)")
    
    if not sys_info['gpus']:
        table.add_row("GPU", "None Detected (using Integrated/CPU)")
        
    console.print(table)
