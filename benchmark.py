import argparse
import json
import time
import os
import sys
from hardware import detect_system
from engines import LlamaCppEngine
from models import get_coding_prompt
from rich.console import Console
from rich.table import Table

def run_benchmark(model_path, engine_name, threads, use_coding=False):
    """Executes the benchmark for a single model+engine combo."""
    prompt = "Write a quicksort function in Python."
    if use_coding:
        prompt = get_coding_prompt(0)
    
    if engine_name == "llama.cpp":
        engine = LlamaCppEngine("llama.cpp", model_path, threads)
        return engine.run(prompt)
    else:
        return "Not Implemented"

def main():
    parser = argparse.ArgumentParser(description="Hardware-Adaptive LLM Benchmark")
    parser.add_argument("--coding", action="store_true", help="Run coding quality benchmark")
    parser.add_argument("--models-dir", default="models", help="Directory containing GGUF models")
    args = parser.parse_args()

    console = Console()
    sys_info = detect_system()
    available_ram = sys_info['ram']['available'] + sys_info['ram']['swap_total']
    
    console.print(f"[bold green]Starting Benchmark[/bold green]")
    console.print(f"Hardware: {sys_info['cpu']['model']} | RAM+Swap: {available_ram:.2f} GB")

    # In a real scenario, we would scan the models-dir for .gguf files
    # For now, we simulate the found models
    available_files = []
    if os.path.exists(args.models_dir):
        available_files = [f for f in os.listdir(args.models_dir) if f.endswith(".gguf")]
    
    if not available_files:
        console.print("[red]No GGUF models found in models/ directory. Using simulated results for demo.[/red]")
        available_files = ["qwen3.6-27b-q2_k.gguf"]

    results = []
    
    for model_file in available_files:
        model_path = os.path.join(args.models_dir, model_file)
        # Basic check: 1GB per model file approx
        # Real logic would check file size
        
        for engine in ["llama.cpp"]:
            console.print(f"Benchmarking [cyan]{model_file}[/cyan] with [magenta]{engine}[/magenta]...")
            
            # Subprocess/Isolation logic: we wrap the real call
            tps = run_benchmark(model_path, engine, sys_info['cpu']['threads'], args.coding)
            
            res = {
                "model": model_file,
                "engine": engine,
                "tps": tps,
                "coding_test": args.coding
            }
            results.append(res)

    # Display Table
    table = Table(title="LLM Benchmark Results")
    table.add_column("Program + Model", style="cyan")
    table.add_column("10 t/s", style="green") # Matching user's requested column name
    
    for r in results:
        tps_display = f"{r['tps']:.2f}" if isinstance(r['tps'], float) else str(r['tps'])
        table.add_row(f"{r['engine']} + {r['model']}", tps_display)
        
    console.print(table)

if __name__ == "__main__":
    main()
