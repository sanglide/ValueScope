"""Generate dataset statistics in the format of the paper figure.

All datasets are now unified under:
  - data/code_scenarios/{handwritten,pipeline,generated}.json
  - data/text_scenarios/issues.json
"""
import json
from pathlib import Path
from collections import Counter

DATA = Path(__file__).parent / "data"

LABEL_NAME = {
    'HV1': 'Conformity', 'HV2': 'Pleasure', 'HV3': 'Dignity', 'HV4': 'Inclusiveness',
    'HV5': 'Sense of Belonging', 'HV6': 'Freedom', 'HV7': 'Independence', 'HV8': 'Wealth',
    'HV9': 'Privacy', 'HV10': 'Security',
    'SV1': 'Trust', 'SV2': 'Correctness', 'SV3': 'Compatibility', 'SV4': 'Portability',
    'SV5': 'Reliability', 'SV6': 'Efficiency', 'SV7': 'Energy Preservation',
    'SV8': 'Usability', 'SV9': 'Accessibility', 'SV10': 'Longevity',
}


def load(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get('samples', [])


code_hand = load(DATA / "code_scenarios" / "handwritten.json")
code_pipe = load(DATA / "code_scenarios" / "pipeline.json")
code_gen = load(DATA / "code_scenarios" / "generated.json")
text_samples = load(DATA / "text_scenarios" / "issues.json")

all_code = code_hand + code_pipe + code_gen
code_pos = sum(1 for s in all_code if s.get("has_value_risk"))
text_pos = sum(1 for s in text_samples if s.get("has_value_risk"))

total = len(all_code) + len(text_samples)
pos = code_pos + text_pos
neg = total - pos

val_counts = Counter()
for s in all_code + text_samples:
    if s.get("has_value_risk"):
        for v in s.get("ground_truth_values", []):
            val_counts[v] += 1

print("Table (a): Summary statistics of the dataset")
print(f"{'Statistical Indicators':<45} {'Value':<20}")
print("-" * 65)
print(f"{'Total sample size':<45} {total}")
print(f"{'Positive instances (with value risks)':<45} {pos} ({pos/total*100:.1f}%)")
print(f"{'Negative instances (without value risks)':<45} {neg} ({neg/total*100:.1f}%)")
print(f"{'Number of value types involved':<45} {len(val_counts)}")

print("\nTable (b): Distribution of human-annotated value labels among positive samples")
print(f"{'ID':<6} {'Value Name':<25} {'Count':<8} {'%':<8}")
print("-" * 65)
for vid in sorted(val_counts.keys()):
    cnt = val_counts[vid]
    name = LABEL_NAME.get(vid, '?')
    pct = cnt / pos * 100
    print(f"{vid:<6} {name:<25} {cnt:<8} {pct:.1f}%")
