# Serve-AI Benchmark

A hardware-adaptive benchmarking suite designed for local execution of Qwen 3.6 models (27B Dense and 35B MoE) on resource-constrained systems.

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
4. **Serve/Chat with a Model Locally:**
   ```bash
   python serve.py
   ```
   (Follow the interactive prompt to select your model).

## Project Structure
- `benchmark.py`: Main entry point and result reporting.
- `hardware.py`: Cross-platform hardware discovery.
- `engines.py`: Integration wrappers for llama.cpp and other servers.
- `models.py`: Configuration for Qwen 3.6 variants and coding prompts.
- `flake.nix`: NixOS development environment.
