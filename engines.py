import time
import os

class BenchmarkEngine:
    def __init__(self, name, model_path, threads=6):
        self.name = name
        self.model_path = model_path
        self.threads = threads

    def run(self, prompt="Write a quicksort function in Python."):
        raise NotImplementedError

class LlamaCppEngine(BenchmarkEngine):
    def __init__(self, name, model_path, threads=6, draft_model_path=None, kv_quant="f16"):
        super().__init__(name, model_path, threads)
        self.draft_model_path = draft_model_path
        self.kv_quant_str = kv_quant.lower()
        
        try:
            import llama_cpp
            self.kv_quant = getattr(llama_cpp, f"GGML_TYPE_{kv_quant.upper()}")
        except ImportError:
            self.kv_quant = 1  # F16 default fallback if module missing
        except AttributeError:
            self.kv_quant = llama_cpp.GGML_TYPE_F16

    def run(self, prompt="Write a quicksort function in Python."):
        try:
            import llama_cpp
            from llama_cpp import Llama
            
            # Using mmap=True is critical for loading large models
            llm = Llama(
                model_path=self.model_path,
                draft_model=self.draft_model_path,
                type_k=self.kv_quant, # KV Cache quantization (TurboQuant style)
                type_v=self.kv_quant,
                flash_attn=True,
                n_threads=self.threads,
                n_ctx=2048,
                verbose=False,
                n_gpu_layers=-1 # Offload all layers to GPU
            )
            
            start = time.time()
            output = llm(prompt, max_tokens=128)
            end = time.time()
            
            elapsed = end - start
            tokens = output['usage']['completion_tokens']
            tps = tokens / elapsed
            text = output['choices'][0]['text']
            return tps, text
        except Exception as e:
            return f"Error: {str(e)}", ""

class TransformersEngine(BenchmarkEngine):
    def __init__(self, name, model_path, threads=6, dtype="float32"):
        super().__init__(name, model_path, threads)
        self.dtype_str = dtype

    def run(self, prompt="Write a quicksort function in Python."):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.bfloat16 if self.dtype_str == "bfloat16" else torch.float32
            
            print(f"    [transformers] Loading model ({self.dtype_str} on {device})...")
            tokenizer_path = self.model_path if os.path.isdir(self.model_path) else "Qwen/Qwen2.5-Coder-32B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                device_map=device_map, 
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            print(f"    [transformers] Generating response...")
            start = time.time()
            outputs = model.generate(**inputs, max_new_tokens=128)
            end = time.time()
            
            elapsed = end - start
            tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
            tps = tokens / elapsed
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return tps, response
        except ImportError:
            return "Error: transformers/torch not installed.", ""
        except Exception as e:
            return f"Error: {str(e)}", ""

class VllmCpuEngine(BenchmarkEngine):
    def run(self, prompt="Write a quicksort function in Python."):
        try:
            return "vLLM-CPU: Manual setup required", ""
        except Exception as e:
            return f"Error: {str(e)}", ""
