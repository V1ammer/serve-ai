# Serve-AI Benchmark

A hardware-adaptive benchmarking suite designed for local execution of Qwen 3.6 models (27B Dense and 35B MoE) on resource-constrained systems.

## Key Features
- **Hardware-Adaptive:** Automatically detects CPU threads, available RAM, and Swap size to determine if a model can safely load.
- **OOM-Safe Architecture:** Uses subprocess isolation for model loading. If the OS OOM Killer (SIGKILL) strikes, the main benchmark script survives and continues to the next test.
- **Cross-Platform:** Full support for NixOS (Flakes), Ubuntu, and Windows.
- **Performance Metrics:** Generates a clean table of Tokens per Second (t/s).
- **Coding Quality:** Includes a `--coding` flag to evaluate models on HumanEval-style Python tasks.

## Supported Engines
- **llama.cpp:** High-efficiency CPU/Vulkan inference (via `llama-cpp-python`).
- **vLLM / SGLang:** Placeholder support for CPU/ROCm backends (requires manual environment setup for APUs).

## Installation

### NixOS (Recommended)
The project includes a `flake.nix` and `.envrc` for a deterministic environment.
```bash
direnv allow
uv sync
```

### Ubuntu
Ensure you have build tools installed:
```bash
sudo apt update && sudo apt install build-essential cmake
uv sync
```

### Windows
1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (C++ workload).
2. Install `uv`.
3. Run `uv sync`.

## Usage
1. **Prepare Models:** Place your `.gguf` files in the `models/` directory.
2. **Run Standard Benchmark:**
   ```bash
   python benchmark.py
   ```
3. **Run Coding Quality Test:**
   ```bash
   python benchmark.py --coding
   ```

## Project Structure
- `benchmark.py`: Main entry point and result reporting.
- `hardware.py`: Cross-platform hardware discovery.
- `engines.py`: Integration wrappers for llama.cpp and other servers.
- `models.py`: Configuration for Qwen 3.6 variants and coding prompts.
- `flake.nix`: NixOS development environment.
