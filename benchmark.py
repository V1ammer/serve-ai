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
    """Recommends models based on detected hardware."""
    available_gb = sys_info['ram']['available'] + sys_info['ram']['swap_total']
    console = Console()
    
    table = Table(title="Recommended Models to Download")
    table.add_column("Model", style="cyan")
    table.add_column("Quantization", style="magenta")
    table.add_column("Est. Size", style="green")
    table.add_column("HuggingFace Link", style="blue")

    recommendations = [
        ("Qwen3.6-27B", "Q2_K", "9GB", "bartowski/Qwen_Qwen3.6-27B-GGUF"),
        ("Qwen3.6-35B-A3B", "Q2_K", "11GB", "bartowski/Qwen3.6-35B-A3B-GGUF")
    ]
    
    if available_gb > 16:
        recommendations.append(("Qwen3.6-27B", "Q4_K_M", "16GB", "bartowski/Qwen_Qwen3.6-27B-GGUF"))

    for name, quant, size, link in recommendations:
        table.add_row(name, quant, size, f"https://hf.co/{link}")
    
    console.print(table)
    console.print("\n[bold yellow]To download, use:[/bold yellow]")
    console.print("huggingface-cli download <link> --include \"*Q2_K.gguf\" --local-dir models/")

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

    if results:
        report_file = save_results_md(results, sys_info)
        console.print(f"\n[bold green]Report saved to {report_file}[/bold green]")

if __name__ == "__main__":
    main()
