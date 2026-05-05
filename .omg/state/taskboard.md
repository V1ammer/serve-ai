# OmG Taskboard

## Goal
Benchmark Qwen3.6-35B-A3B (MoE) and Qwen3.6-27B (Dense) on constrained hardware (14GB RAM, AMD APU) using llama.cpp, vLLM, and sglang, producing a t/s performance table and optional coding quality evaluation.

## Non-Goals
- Full fine-tuning or model training.
- Benchmarking on non-requested hardware (e.g., discrete Nvidia GPUs).
- Running unquantized models that physically exceed 14GB RAM.

## Risks & Mitigations
- **Risk:** 14GB RAM is severely limiting for 35B and 27B models.
  - **Mitigation:** Use aggressive GGUF quantization (e.g., 2-bit, 3-bit, 4-bit) and dynamic RAM detection to skip models that cannot load.
- **Risk:** vLLM and sglang might struggle or lack robust support for CPU/APU execution compared to llama.cpp.
  - **Mitigation:** Ensure fallback to CPU execution modes, handle engine initialization failures gracefully, and continue to the next configuration.

## Tasks

| ID | Priority | Status | Title | Lane | Dependencies |
| :--- | :--- | :--- | :--- | :--- | :--- |
| T01 | p0 | done | Git Setup (Init, Remote, First Commit) | `omg-executor` | |
| T02 | p0 | done | `uv` Project Initialization | `omg-executor` | T01 |
| T03 | p0 | ready | Configure Dependencies (Nix Flake setup) | `omg-executor` | T02 |
| T04 | p1 | done | Implement Hardware Detection Logic | `omg-executor` | T03 |
| T05 | p1 | ready | Implement Model Selection & GGUF Quant Mapping | `omg-executor` | T04 |
| T06 | p1 | blocked | Implement Engine Integration (CPU/APU limits) | `omg-executor` | T05 |
| T07 | p1 | done | Implement Iterative Execution & t/s Table Output | `omg-executor` | T06 |
| T08 | p2 | blocked | Integrate Coding Quality Benchmark (e.g., HumanEval) | `omg-executor` | T07 |
| T09 | p0 | blocked | System Verification & End-to-End Test | `omg-verifier` | T07, T08 |

## Phase Plan
1. **Setup:** T01, T02, T03. Get the repository, package manager (`uv`), and dependency foundation ready.
2. **Infrastructure Logic:** T04, T05. Detect hardware limits (14GB) and map the requested Qwen3.6 models to viable quantized GGUF formats.
3. **Engine Implementation:** T06. Abstract the execution engines (llama.cpp, vLLM, sglang), specifically handling CPU/APU constraints and skipping on OOM.
4. **Benchmarking & Reporting:** T07, T08. Create the main loop to run models across engines, measure tokens/sec, execute coding evaluations, and print the result table.
5. **Validation:** T09. Ensure the script runs end-to-end handling OOM limits gracefully.

## Critical Files for Implementation
1. `pyproject.toml` (managed by uv)
2. `benchmark.py` (main loop and table output)
3. `hardware.py` (RAM/VRAM detection)
4. `models.py` (quantization mapping logic)
5. `engines.py` (llama.cpp, vLLM, sglang wrappers)
