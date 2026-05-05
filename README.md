# Serve-AI Benchmark

Hardware-adaptive benchmarking for Qwen 3.6 models.

## Support
- **NixOS:** Use the provided `flake.nix` and `direnv`.
- **Ubuntu:** Install `build-essential cmake` and use `uv`.
- **Windows:** Install `VS Build Tools` (C++) and use `uv`.

## Setup
1. **Python Dependencies:**
   ```bash
   uv sync
   ```

2. **Model Download:**
   Download Qwen 3.6 GGUF (Q2_K) files to the `models/` directory.

3. **Run Benchmark:**
   ```bash
   python benchmark.py
   ```

## Design
- **Subprocess Isolation:** Models are loaded in child processes to prevent OOM from killing the main benchmark script.
- **Hardware-Adaptive:** Automatically detects RAM/Swap and skips models that won't fit.
- **Incremental Saving:** Results are saved to `results.jsonl` after each run.
