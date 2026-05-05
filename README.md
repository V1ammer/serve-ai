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
1. **Find Best Models for Your Hardware:**
   ```bash
   uv run benchmark.py --recommend
   ```
   (This will show you exactly which Qwen 3.6 versions will fit in your RAM and give you download links).

2. **Prepare Models:** Place your `.gguf` files in the `models/` directory.

3. **Run Standard Benchmark:**
   ```bash
   uv run benchmark.py
   ```

4. **Run Coding Quality Test:**
   ```bash
   uv run benchmark.py --coding
   ```

5. **Serve/Chat with a Model Locally:**
   ```bash
   uv run serve.py
   ```
   (Follow the interactive prompt to select your model).

## Project Structure
- `benchmark.py`: Main entry point and result reporting.
- `hardware.py`: Cross-platform hardware discovery.
- `engines.py`: Integration wrappers for llama.cpp and other servers.
- `models.py`: Configuration for Qwen 3.6 variants and coding prompts.
- `flake.nix`: NixOS development environment.
