# Architecture Review: serve-ai

## 1. Hardware Feasibility Analysis
**Constraints:** 14GB Total RAM, ~7GB Available. AMD Ryzen 5 5600H (12 threads). No dGPU.

*   **Qwen3.6-35B-A3B (MoE):**
    *   35B MoE typically requires high memory. A MoE at 35B means roughly 11B active parameters.
    *   At 2-bit (Q2_K GGUF), the file size is ~10-12GB.
    *   **Feasibility:** HIGH RISK / LIKELY OOM. Loading this requires heavily paging to swap, which will cause the OS to thrash and t/s to drop significantly. 7GB available RAM is not enough to keep the active parameters in memory.
*   **Qwen3.6-27B (Dense):**
    *   At 2-bit (Q2_K GGUF), the file size is ~8-9GB.
    *   At 3-bit (Q3_K_M GGUF), file size is ~11GB.
    *   **Feasibility:** MODERATE RISK. The 2-bit version might barely fit if the OS frees up caches, but with 7GB currently available, it will still page to swap.

*Recommendation:* Use `mmap` aggressively with `llama.cpp` to keep inactive layers on disk. Set OS swapiness high or ensure NVMe swap is active.

## 2. Engine Configurations
*   **llama.cpp:**
    *   **Backend:** CPU + Vulkan (for AMD APU).
    *   **Flags:** `--threads 6` (leaving 6 for OS/background to prevent starvation), `--mmap 1` (essential for loading models > available RAM), `--mlock 0`, `-ngl <num>` (offload some layers to APU memory via Vulkan, but APU shares the same RAM, so monitor system RAM closely, perhaps start with `-ngl 10`).
*   **vLLM:**
    *   **Backend:** OpenVINO or Neural Magic CPU.
    *   **Feasibility:** NOT RECOMMENDED. vLLM uses large Python/PyTorch overheads and allocates large KV caches upfront. It will almost certainly OOM on 7GB available RAM before even loading the model. If attempted, use `gpu_memory_utilization=0.4` or enforce extreme KV cache limits (`--max-num-seqs 1`).
*   **SGLang:**
    *   **Backend:** CPU.
    *   **Feasibility:** NOT RECOMMENDED for the same reasons as vLLM. Python-based serving frameworks are memory-heavy.

*Recommendation:* Focus 90% of effort on `llama.cpp` using Python bindings (`llama-cpp-python` or direct subprocess calls). Fallback engines are highly likely to fail on memory constraints.

## 3. Benchmark Script Architecture
To prevent OOM crosstalk and dangling memory:
*   **Process Isolation:** The main script (`benchmark.py`) must spawn each engine run as a separate subprocess (e.g., using `subprocess.run`). Do not instantiate models in the main Python process's memory space.
*   **Memory Teardown:** After each subprocess finishes, run a cleanup routine to free OS pagecache if possible (e.g., `sync` or rely on the subprocess death to reclaim memory).
*   **OOM Handling:** The OS OOM Killer is highly likely to strike. The main script must catch non-zero exit codes (specifically 137 for SIGKILL) from the subprocess, mark that config as "FAILED (OOM)", and gracefully continue to the next model/engine combination.
*   **Progress Tracking:** Use a JSON Lines (JSONL) file or SQLite to log results as they complete, so a crash doesn't wipe out the whole benchmark run.

## 4. Risks in "Coding Quality Benchmark" (HumanEval)
*   **Generation Time:** HumanEval requires generating 164 samples. At ~1 t/s (swapping heavily), one sample (average 200 tokens) takes over 3 minutes. 164 samples = 8+ hours.
*   **Context Limit:** The KV cache will consume additional RAM. Keep max context (`-c`) small, e.g., 2048 or 4096 tokens, to save RAM for the model weights.
*   **Timeout:** The subprocess needs a very high timeout. Code execution for HumanEval itself (if run locally) is fast, but the generation phase will be extremely slow.

## Implementation Handoff Checklist
- [ ] **omg-planner:** Update the taskboard to reflect that vLLM and SGLang are low-priority/stretch goals. Focus the critical path on `llama.cpp`.
- [ ] **omg-executor:** Create `hardware.py` to assert swap file size and available RAM before attempting to load a model.
- [ ] **omg-executor:** Structure `benchmark.py` to use `subprocess` for all engine instantiations (isolated process architecture).
- [ ] **omg-executor:** Target `Q2_K` model variants first for testing to ensure the pipeline functions before trying larger/denser models.
- [ ] **omg-executor:** Implement OOM-killer detection (Exit Code 137) in the benchmark loop.