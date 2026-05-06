import os
import argparse
from hardware import detect_system
from engines import LlamaCppEngine
from rich.console import Console

def main():
    parser = argparse.ArgumentParser(description="Serve-AI: Local LLM Runner")
    parser.add_argument("--model", help="Path to the model file")
    parser.add_argument("--draft", help="Path to the draft model (DFlash) for speedup")
    parser.add_argument("--kv-quant", default="q8_0", help="KV Cache quantization type (e.g., q4_0, q8_0) for TurboQuant-like effect")
    parser.add_argument("--engine", default="llama.cpp", choices=["llama.cpp", "vllm", "transformers"], help="Engine to use")
    args = parser.parse_args()

    console = Console()
    sys_info = detect_system()
    
    models_dir = "models"
    available_models = []
    if os.path.exists(models_dir):
        available_models = [f for f in os.listdir(models_dir) if f.endswith((".gguf", ".safetensors"))]

    selected_model = args.model

    if not selected_model:
        if not available_models:
            console.print("[red]No models found in models/ directory.[/red]")
            return
        
        console.print("[bold cyan]Available Models:[/bold cyan]")
        for i, m in enumerate(available_models):
            console.print(f"{i+1}. {m}")
        
        choice = input("\nSelect a model (number) or path: ")
        try:
            idx = int(choice) - 1
            selected_model = os.path.join(models_dir, available_models[idx])
        except (ValueError, IndexError):
            selected_model = choice

    if not os.path.exists(selected_model):
        console.print(f"[red]Model file not found: {selected_model}[/red]")
        return

    # Auto-detect engine if not explicitly forced, or if forced but incompatible
    if selected_model.endswith(".safetensors") and args.engine == "llama.cpp":
        console.print("[yellow]Note: .safetensors files require transformers engine. Switching to transformers.[/yellow]")
        args.engine = "transformers"
    elif selected_model.endswith(".gguf") and args.engine == "transformers":
        console.print("[yellow]Note: .gguf files require llama.cpp engine. Switching to llama.cpp.[/yellow]")
        args.engine = "llama.cpp"

    if args.engine == "llama.cpp":
        engine = LlamaCppEngine("llama.cpp", selected_model, sys_info['cpu']['threads'], args.draft, args.kv_quant)
        engine.chat()
    elif args.engine == "transformers":
        engine = TransformersEngine("transformers", selected_model, sys_info['cpu']['threads'])
        engine.chat()
    else:
        console.print("[yellow]vLLM serving for APU/CPU requires specific configuration (ROCm/OpenVINO).[/yellow]")
        console.print("Currently, only llama.cpp is optimized for 14GB RAM local chat.")

if __name__ == "__main__":
    main()
