MODELS_CONFIG = {
    "qwen3.6-27b": {
        "full_name": "Qwen3.6-27B",
        "variants": ["Q2_K", "Q3_K_L", "Q4_K_M"],
        "size_gb_est": {"Q2_K": 9, "Q3_K_L": 11, "Q4_K_M": 15}
    },
    "qwen3.6-35b-moe": {
        "full_name": "Qwen3.6-35B-A3B (MoE)",
        "variants": ["Q2_K", "Q3_K_L"],
        "size_gb_est": {"Q2_K": 11, "Q3_K_L": 14}
    }
}

CODING_BENCHMARK_PROMPTS = [
    {
        "id": "humaneval_0",
        "task": "Check if a list of numbers has any two elements that are closer to each other than a given threshold.",
        "signature": "def has_close_elements(numbers: List[float], threshold: float) -> bool:"
    },
    {
        "id": "humaneval_1",
        "task": "Separate a string of parentheses into groups.",
        "signature": "def separate_paren_groups(paren_string: str) -> List[str]:"
    }
]

def get_coding_prompt(index=0):
    item = CODING_BENCHMARK_PROMPTS[index % len(CODING_BENCHMARK_PROMPTS)]
    return f"Task: {item['task']}\n\n{item['signature']}"
