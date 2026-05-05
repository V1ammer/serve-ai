CODING_BENCHMARK_PROMPTS = [
    {
        "id": "humaneval_0",
        "task": "Check if a list of numbers has any two elements that are closer to each other than a given threshold.",
        "signature": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:",
        "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\ncheck(has_close_elements)"
    },
    {
        "id": "humaneval_1",
        "task": "Separate a string of parentheses into groups.",
        "signature": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:",
        "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\ncheck(separate_paren_groups)"
    }
]

def get_coding_task(index=0):
    return CODING_BENCHMARK_PROMPTS[index % len(CODING_BENCHMARK_PROMPTS)]
