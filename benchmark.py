import argparse
import time
import os
import sys
import re
from hardware import detect_system
from engines import LlamaCppEngine, TransformersEngine
from rich.console import Console
from rich.table import Table

def estimate_max_context(model_path, fast_mem_gb, kv_bits=16):
    """Heuristic estimation of maximum context window based on fast memory and quantization."""
    try:
        model_size_gb = os.path.getsize(model_path) / (1024**3)
    except OSError:
        model_size_gb = 15.0 # fallback for estimated size
        
    # 1.5GB OS Headroom
    kv_mem_avail_gb = fast_mem_gb - model_size_gb - 1.5
    if kv_mem_avail_gb <= 0:
        return 0
        
    # Rough heuristic: A 27B-35B model uses ~160KB per token at fp16 (16 bits)
    bytes_per_token = 160000 * (kv_bits / 16.0)
    tokens = int((kv_mem_avail_gb * 1024**3) / bytes_per_token)
    
    # Cap at standard long context size
    return min(tokens, 131072)

def format_context_size(ctx_len):
    if ctx_len <= 0:
        return "OOM"
    elif ctx_len >= 1000:
        return f"{ctx_len // 1000}k"
    return str(ctx_len)

def evaluate_code(generated_text, signature, test_code):
    """Evaluates the generated code against the provided tests."""
    code = generated_text
    match = re.search(r"```(?:python)?(.*?)```", generated_text, re.DOTALL)
    if match:
        code = match.group(1).strip()
    
    func_name_match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", signature)
    if func_name_match:
        func_name = func_name_match.group(1)
        if func_name not in code:
            code = signature + "\n" + code

    full_code = "from typing import List, Dict, Tuple, Optional\n" + code + "\n\n" + test_code
    
    try:
        exec(full_code, {})
        return "[green]Pass[/green]"
    except AssertionError:
        return "[red]Fail (Assertion)[/red]"
    except Exception as e:
        return f"[red]Fail ({type(e).__name__})[/red]"

def run_benchmark(model_path, config, threads, use_coding=True):
    """Executes the benchmark for a single config matrix."""
    task = None
    if use_coding:
        from models import get_coding_task
        task = get_coding_task(0)
        prompt = f"Task: {task['task']}\n\n{task['signature']}\n"
    else:
        prompt = "Write a quicksort function in Python."
    
    if config["engine"] == "llama.cpp":
        engine = LlamaCppEngine("llama.cpp", model_path, threads, kv_quant=config.get("kv_quant", "f16"))
        res = engine.run(prompt)
    elif config["engine"] == "transformers":
        engine = TransformersEngine("transformers", model_path, threads, dtype=config.get("dtype", "float32"))
        res = engine.run(prompt)
    else:
        res = ("Not Implemented", "")
        
    tps, text = res if isinstance(res, tuple) else (res, "")
    
    quality = "N/A"
    if use_coding and isinstance(tps, float):
        quality = evaluate_code(text, task['signature'], task['test'])
        
    return tps, quality

def save_results_md(results, sys_info):
    """Saves matrix benchmark results to a Markdown file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    filename = f"benchmark_report_{int(time.time())}.md"
    
    has_coding = any(r.get('coding_test') for r in results)
    
    with open(filename, "w") as f:
        f.write(f"# Serve-AI Benchmark Report\n\n")
        f.write(f"**Date:** {timestamp}  \n")
        f.write(f"**Hardware:** {sys_info['cpu']['model']} | RAM: {sys_info['ram']['total']:.2f} GB  \n\n")
        
        if has_coding:
            f.write("| Program (Config) | Model | t/s | Max Ctx | Quality |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        else:
            f.write("| Program (Config) | Model | t/s | Max Ctx |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            
        for r in results:
            tps_display = f"{r['tps']:.2f}" if isinstance(r['tps'], float) else str(r['tps'])
            ctx_display = format_context_size(r['max_ctx'])
            
            if has_coding:
                quality_md = re.sub(r'\[/?(?:red|green|yellow|cyan|magenta|bold|italic|underline)\]', '', r.get('quality', 'N/A'))
                f.write(f"| {r['config_name']} | {r['model']} | {tps_display} | {ctx_display} | {quality_md} |\n")
            else:
                f.write(f"| {r['config_name']} | {r['model']} | {tps_display} | {ctx_display} |\n")
            
    return filename

def recommend_models(sys_info, models_dir="models"):
    """Recommends best possible quantization based on total fast memory."""
    # Fast memory only: RAM + ZRAM + VRAM
    total_vram = sum(gpu.get('vram_total', 0) for gpu in sys_info['gpus'])
    total_fast_mem = sys_info['ram']['total'] + sys_info['ram']['zram'] + total_vram
    
    # OS headroom: 15% or 1.5GB
    safe_limit = max(total_fast_mem * 0.85, total_fast_mem - 1.5)
    
    # Check existing files
    existing_files = []
    if os.path.exists(models_dir):
        existing_files = [f.lower() for f in os.listdir(models_dir) if ".gguf" in f.lower() or ".safetensors" in f.lower()]

    console = Console()
    table = Table(title=f"Hardware-Adaptive Recommendations (Fast Memory: {total_fast_mem:.1f}GB)")
    table.add_column("Model Family", style="cyan")
    table.add_column("Best Quant for You", style="magenta")
    table.add_column("Est. Size", style="green")
    table.add_column("Quality Level", style="yellow")
    table.add_column("Status", style="bold")
    table.add_column("HuggingFace Link", style="blue")

    # Full standard GGUF quantization map (approximate bits per weight)
    QUANTS = [
        ("Q8_0", 8.5, "Near Lossless"),
        ("Q6_K", 6.6, "Very High Quality"),
        ("Q5_K_M", 5.7, "High (Medium)"),
        ("Q5_K_S", 5.5, "High (Small)"),
        ("Q4_K_M", 4.8, "Standard (Medium)"),
        ("Q4_K_S", 4.6, "Standard (Small)"),
        ("Q3_K_L", 4.2, "Light (Large)"),
        ("Q3_K_M", 3.9, "Light (Medium)"),
        ("Q3_K_S", 3.5, "Light (Small)"),
        ("Q2_K", 3.3, "Extreme Compression"),
        ("IQ2_M", 2.7, "Ultra (Medium)"),
        ("IQ2_XS", 2.3, "Ultra (Extra Small)")
    ]

    MODELS_TO_CHECK = [
        ("Qwen3.6-27B", 27, "unsloth/Qwen3.6-27B-GGUF"),
        ("Qwen3.6-35B-A3B", 35, "unsloth/Qwen3.6-35B-A3B-GGUF")
    ]

    for name, params, link in MODELS_TO_CHECK:
        viable_quants = []
        for q_name, bits, desc in QUANTS:
            est_size = (params * bits) / 8
            if est_size < safe_limit:
                viable_quants.append((q_name, est_size, desc))
        
        if viable_quants:
            for i, (q_name, size, desc) in enumerate(viable_quants[:3]):
                label = f"{name}" if i == 0 else ""
                
                # Check status
                status = "[red]Missing[/red]"
                if any(name.lower() in f and q_name.lower() in f for f in existing_files):
                    status = "[green]Downloaded[/green]"
                
                table.add_row(
                    label, q_name, f"{size:.1f}GB", desc, status, 
                    f"https://hf.co/{link}" if i == 0 else ""
                )
            
            # Suggest DFlash drafters
            drafter_link = link.replace("unsloth/", "z-lab/").replace("-GGUF", "-DFlash")
            
            if "35B" in name:
                drafter_quants = [("Safetensors/BF16", "~1.5GB", "")]
            else:
                drafter_quants = [("Q8_0", "<1.5GB", "q8_0"), ("Q4_K_M", "<0.8GB", "q4_k_m")]
                
            for dq_label, dsize, dq_search in drafter_quants:
                drafter_status = "[red]Missing[/red]"
                if any("dflash" in f and name.lower() in f and (not dq_search or dq_search in f.lower()) for f in existing_files):
                    drafter_status = "[green]Downloaded[/green]"
                
                table.add_row(
                    "", f"[bold yellow]DFlash Drafter ({dq_label})[/bold yellow]", dsize, "Speedup 2x-4x", 
                    drafter_status, f"https://hf.co/{drafter_link}"
                )
        else:
            table.add_row(name, "None", "Too Large", "N/A", "[grey]N/A[/grey]", f"https://hf.co/{link}")
    
    console.print(table)
    console.print("\n[bold yellow]Selection Logic:[/bold yellow] Chosen based on 80% of your total RAM+VRAM+Swap.")

def main():
    parser = argparse.ArgumentParser(description="Hardware-Adaptive LLM Benchmark Matrix")
    parser.add_argument("--recommend", action="store_true", help="Show recommended models for your hardware")
    parser.add_argument("--models-dir", default="models", help="Directory containing models")
    args = parser.parse_args()

    console = Console()
    sys_info = detect_system()
    
    if args.recommend:
        recommend_models(sys_info, args.models_dir)
        return
        
    available_fast_mem = sys_info['ram']['available'] + sys_info['ram']['zram']
    
    console.print(f"[bold green]Starting Matrix Benchmark[/bold green]")
    console.print(f"Hardware: {sys_info['cpu']['model']} | Fast Memory (Avail+ZRAM): {available_fast_mem:.2f} GB\n")

    available_files = []
    if os.path.exists(args.models_dir):
        available_files = [f for f in os.listdir(args.models_dir) if f.endswith((".gguf", ".safetensors")) and "dflash" not in f.lower()]

    if not available_files:
        console.print("[red]No models found in models/ directory. Run with valid files.[/red]")
        return

    results = []
    
    for model_file in available_files:
        model_path = os.path.join(args.models_dir, model_file)
        
        configs_to_run = []
        if model_file.endswith(".gguf"):
            configs_to_run.extend([
                {"name": "llama.cpp (F16 KV)", "engine": "llama.cpp", "kv_quant": "f16", "kv_bits": 16},
                {"name": "llama.cpp (Q8_0 TurboQuant)", "engine": "llama.cpp", "kv_quant": "q8_0", "kv_bits": 8},
                {"name": "llama.cpp (Q4_0 TurboQuant)", "engine": "llama.cpp", "kv_quant": "q4_0", "kv_bits": 4},
            ])
        if model_file.endswith(".safetensors"):
            configs_to_run.extend([
                {"name": "transformers (FP32)", "engine": "transformers", "dtype": "float32", "kv_bits": 32},
                {"name": "transformers (BF16)", "engine": "transformers", "dtype": "bfloat16", "kv_bits": 16},
            ])
            
        for config in configs_to_run:
            console.print(f"▶ Benchmarking [cyan]{model_file}[/cyan] with [magenta]{config['name']}[/magenta]...")
            console.print(f"  [dim]↳ Path: {model_path}[/dim]")
            
            start_run = time.time()
            tps, quality = run_benchmark(model_path, config, sys_info['cpu']['threads'], True)
            end_run = time.time()
            
            tps_str = f"{tps:.2f}" if isinstance(tps, float) else str(tps)
            console.print(f"  [green]✔ Completed in {end_run - start_run:.2f}s[/green] | TPS: {tps_str} | Quality: {quality}\n")
            
            max_ctx = estimate_max_context(model_path, available_fast_mem, config.get("kv_bits", 16))
            
            res = {
                "model": model_file,
                "config_name": config["name"],
                "tps": tps,
                "quality": quality,
                "max_ctx": max_ctx,
                "coding_test": True
            }
            results.append(res)

    # Display Table
    print("")
    table = Table(title="LLM Matrix Benchmark Results")
    table.add_column("Program (Config)", style="magenta")
    table.add_column("Model", style="cyan")
    table.add_column("t/s", style="green")
    table.add_column("Max Ctx", style="blue")
    table.add_column("Quality", style="yellow")
    
    for r in results:
        tps_display = f"{r['tps']:.2f}" if isinstance(r['tps'], float) else str(r['tps'])
        ctx_display = format_context_size(r['max_ctx'])
        row = [r['config_name'], r['model'], tps_display, ctx_display, r['quality']]
        table.add_row(*row)
        
    console.print(table)

    if results:
        report_file = save_results_md(results, sys_info)
        console.print(f"\n[bold green]Report saved to {report_file}[/bold green]")

if __name__ == "__main__":
    main()
