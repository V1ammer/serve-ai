import time
import subprocess
from llama_cpp import Llama

class BenchmarkEngine:
    def __init__(self, name, model_path, threads=6):
        self.name = name
        self.model_path = model_path
        self.threads = threads

    def run(self, prompt="Write a quicksort function in Python."):
        raise NotImplementedError

class LlamaCppEngine(BenchmarkEngine):
    def run(self, prompt="Write a quicksort function in Python."):
        try:
            # Using mmap=True is critical for 14GB RAM systems loading large models
            llm = Llama(
                model_path=self.model_path,
                n_threads=self.threads,
                n_ctx=2048,
                verbose=False,
                n_gpu_layers=0 # Force CPU/APU via system RAM
            )
            
            start = time.time()
            output = llm(prompt, max_tokens=128)
            end = time.time()
            
            elapsed = end - start
            tokens = output['usage']['completion_tokens']
            tps = tokens / elapsed
            return tps
        except Exception as e:
            return f"Error: {str(e)}"

    def chat(self):
        """Interactive chat session for local use."""
        print(f"Loading {self.model_path} for chat...")
        try:
            llm = Llama(
                model_path=self.model_path,
                n_threads=self.threads,
                n_ctx=2048,
                verbose=False
            )
            print("\nModel loaded! Type '/exit' to quit.\n")
            while True:
                user_input = input("User: ")
                if user_input.lower() in ["/exit", "/quit"]:
                    break
                
                print("Assistant: ", end="", flush=True)
                for chunk in llm(user_input, stream=True, max_tokens=512):
                    text = chunk['choices'][0]['text']
                    print(text, end="", flush=True)
                print("\n")
        except Exception as e:
            print(f"Chat Error: {e}")

class VllmCpuEngine(BenchmarkEngine):
    def run(self, prompt="Write a quicksort function in Python."):
        # vLLM on CPU is heavy. We use a subprocess to isolate its memory usage completely.
        # This prevents it from killing the main benchmark script.
        # Note: Requires vllm installed via uv
        try:
            # Simulated subprocess command - would be: python -m vllm.entrypoints.api_server ...
            return "vLLM-CPU: Manual setup required for ROCm/APU"
        except Exception as e:
            return f"Error: {str(e)}"
