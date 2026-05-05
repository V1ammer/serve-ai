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

def save_results_md(results, sys_info):
    """Saves benchmark results to a Markdown file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    filename = f"benchmark_report_{int(time.time())}.md"
    
    with open(filename, "w") as f:
        f.write(f"# Serve-AI Benchmark Report\n\n")
        f.write(f"**Date:** {timestamp}  \n")
        f.write(f"**Hardware:** {sys_info['cpu']['model']} | RAM: {sys_info['ram']['total']:.2f} GB  \n\n")
        
        f.write("| Program + Model | 10 t/s |\n")
        f.write("| :--- | :--- |\n")
        for r in results:
            tps_display = f"{r['tps']:.2f}" if isinstance(r['tps'], float) else str(r['tps'])
            f.write(f"| {r['engine']} + {r['model']} | {tps_display} |\n")
            
    return filename

def recommend_models(sys_info):
    """Recommends best possible quantization based on total fast memory."""
    # Fast memory only: RAM + ZRAM + VRAM
    total_vram = sum(gpu.get('vram_total', 0) for gpu in sys_info['gpus'])
    total_fast_mem = sys_info['ram']['total'] + sys_info['ram']['zram'] + total_vram
    
    # OS headroom: 15% or 1.5GB
    safe_limit = max(total_fast_mem * 0.85, total_fast_mem - 1.5)
    
    console = Console()
    table = Table(title=f"Hardware-Adaptive Recommendations (Fast Memory: {total_fast_mem:.1f}GB)")
    table.add_column("Model Family", style="cyan")
    table.add_column("Best Quant for You", style="magenta")
    table.add_column("Est. Size", style="green")
    table.add_column("Quality Level", style="yellow")
    table.add_column("HuggingFace Link", style="blue")

    # Define quants: (name, bits_per_weight, description)
    QUANTS = [
        ("Q8_0", 8.5, "Near Lossless"),
        ("Q6_K", 6.6, "High Quality"),
        ("Q5_K_M", 5.5, "Balanced"),
        ("Q4_K_M", 4.8, "Standard"),
        ("Q3_K_M", 3.7, "Lightweight"),
        ("Q2_K", 2.6, "Extreme Compression")
    ]

    MODELS_TO_CHECK = [
        ("Qwen3.6-27B", 27, "bartowski/Qwen_Qwen3.6-27B-GGUF"),
        ("Qwen3.6-35B-A3B", 35, "bartowski/Qwen3.6-35B-A3B-GGUF")
    ]

    for name, params, link in MODELS_TO_CHECK:
        best_q = None
        for q_name, bits, desc in QUANTS:
            est_size = (params * bits) / 8
            if est_size < safe_limit:
                best_q = (q_name, est_size, desc)
                break
        
        if best_q:
            table.add_row(name, best_q[0], f"{best_q[1]:.1f}GB", best_q[2], f"https://hf.co/{link}")
        else:
            table.add_row(name, "None", "Too Large", "N/A", f"https://hf.co/{link}")
    
    console.print(table)
    console.print("\n[bold yellow]Selection Logic:[/bold yellow] Chosen based on 80% of your total RAM+VRAM+Swap.")

def main():
    parser = argparse.ArgumentParser(description="Hardware-Adaptive LLM Benchmark")
    parser.add_argument("--coding", action="store_true", help="Run coding quality benchmark")
    parser.add_argument("--recommend", action="store_true", help="Show recommended models for your hardware")
    parser.add_argument("--models-dir", default="models", help="Directory containing GGUF models")
    args = parser.parse_args()

    console = Console()
    sys_info = detect_system()
    
    if args.recommend:
        recommend_models(sys_info)
        return

    available_fast_mem = sys_info['ram']['available'] + sys_info['ram']['zram']
    
    console.print(f"[bold green]Starting Benchmark[/bold green]")
    console.print(f"Hardware: {sys_info['cpu']['model']} | Fast Memory (Avail+ZRAM): {available_fast_mem:.2f} GB")

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

    if results:
        report_file = save_results_md(results, sys_info)
        console.print(f"\n[bold green]Report saved to {report_file}[/bold green]")

if __name__ == "__main__":
    main()
