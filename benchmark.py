import subprocess
import json
import time
import os
import sys
from hardware import detect_system
from rich.console import Console
from rich.table import Table

# Configuration
MODELS = [
    {"name": "Qwen3.6-27B-Q2_K", "path": "models/qwen3.6-27b-q2_k.gguf", "size_gb": 9},
    {"name": "Qwen3.6-35B-A3B-Q2_K", "path": "models/qwen3.6-35b-moe-q2_k.gguf", "size_gb": 11},
]

ENGINES = ["llama.cpp", "vLLM (CPU)"]

RESULTS_FILE = "results.jsonl"

def run_llama_cpp(model_path, threads):
    """Simplified subprocess call to llama.cpp (simulated for logic)"""
    # In a real scenario, this would call ./llama-cli or a python wrapper
    # We use a placeholder command to demonstrate the subprocess isolation
    cmd = [
        "python", "-c", 
        f"import time; print('TOKEN_STREAM_START'); time.sleep(2); print('TOKENS_PER_SEC: 12.5')"
    ]
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if process.returncode == 0:
            for line in process.stdout.split("\n"):
                if "TOKENS_PER_SEC:" in line:
                    return float(line.split(":")[1].strip())
        return None
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        if getattr(e, 'returncode', None) == 137:
            return "OOM"
        return str(e)

def benchmark_loop():
    console = Console()
    sys_info = detect_system()
    available_ram = sys_info['ram']['available']
    
    console.print(f"[bold green]Starting Benchmark[/bold green] (Available RAM: {available_ram:.2f} GB)")
    
    results = []

    for model in MODELS:
        # Hardware-adaptive check
        if model['size_gb'] > (available_ram + sys_info['ram']['swap_total']) * 0.9:
            console.print(f"[yellow]Skipping {model['name']}: Insufficient RAM+Swap[/yellow]")
            continue
            
        for engine in ENGINES:
            console.print(f"Running [cyan]{model['name']}[/cyan] on [magenta]{engine}[/magenta]...")
            
            # Here we would branch logic based on engine
            # For this MVP, we simulate the llama.cpp call
            tps = run_llama_cpp(model['path'], sys_info['cpu']['threads'])
            
            res = {
                "model": model['name'],
                "engine": engine,
                "tps": tps,
                "timestamp": time.time()
            }
            results.append(res)
            
            # Incremental save
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(res) + "\n")
                
    return results

def display_results(results):
    console = Console()
    table = Table(title="Benchmark Results")
    table.add_column("Model + Engine", style="cyan")
    table.add_column("Tokens/sec", style="green")
    
    for r in results:
        tps_str = str(r['tps']) if isinstance(r['tps'], (float, int)) else f"[red]{r['tps']}[/red]"
        table.add_row(f"{r['model']} ({r['engine']})", tps_str)
        
    console.print(table)

if __name__ == "__main__":
    results = benchmark_loop()
    display_results(results)
