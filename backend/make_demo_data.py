"""Create a labeled synthetic dataset for demonstrating the complete workflow."""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUMAN = [
    "def solve(values):\n    out = []\n    for value in values:\n        if value % 2 == 0:\n            out.append(value)\n    return out\n",
    "def solve(values):\n    # keep the even values\n    result=[]\n    for n in values:\n        if n%2==0: result += [n]\n    return result\n",
    "def solve(values):\n    evens = []\n    i = 0\n    while i < len(values):\n        current = values[i]\n        if current % 2 == 0:\n            evens.append(current)\n        i += 1\n    return evens\n",
    "def solve(values):\n    picked = []\n    for item in values:\n        even = item % 2 == 0\n        if even:\n            picked.append(item)\n    return picked\n",
]
AI = [
    '"""Return all even integers from the input sequence."""\ndef solve(values):\n    """Filter values according to their parity."""\n    return [value for value in values if value % 2 == 0]\n',
    "def solve(values):\n    \"\"\"Select every element divisible by two.\"\"\"\n    filtered_values = [number for number in values if number % 2 == 0]\n    return filtered_values\n",
    "def solve(values):\n    \"\"\"Build and return a list containing even values.\"\"\"\n    even_values = []\n    for value in values:\n        if value % 2 == 0:\n            even_values.append(value)\n    return even_values\n",
    "def solve(values):\n    \"\"\"Return values whose remainder after division by two is zero.\"\"\"\n    return list(filter(lambda item: item % 2 == 0, values))\n",
]

def build():
    random.seed(42)
    output = []
    for problem in range(1, 21):
        for label, templates in (("human", HUMAN), ("ai", AI)):
            for variant in range(5):
                code = templates[(problem + variant) % len(templates)]
                code = code.replace("% 2", f"% {problem + 1}")
                code = code.replace("def solve(", f"def solve_{problem}(").replace("def process(", f"def process_{problem}(")
                # Harmless, deterministic naming variation keeps samples distinct.
                if variant == 1:
                    code = code.replace("values", "items")
                elif variant == 2:
                    code = code.replace("solve", "process")
                elif variant == 3:
                    code = code.replace("return ", "return ") + "\n"
                output.append({"code": code, "label": label, "problem_id": f"problem-{problem:02d}"})
    random.shuffle(output)
    target = ROOT / "data" / "samples.csv"
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "label", "problem_id"])
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} synthetic examples to {target}")

if __name__ == "__main__":
    build()
