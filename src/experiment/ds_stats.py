import json
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent / 'data'

LABEL_NAME = {
    'HV1': 'Conformity',       'HV2': 'Pleasure',        'HV3': 'Dignity',
    'HV4': 'Inclusiveness',    'HV5': 'Sense of Belonging', 'HV6': 'Freedom',
    'HV7': 'Independence',     'HV8': 'Wealth',          'HV9': 'Privacy',
    'HV10': 'Security',
    'SV1': 'Trust',            'SV2': 'Correctness',     'SV3': 'Compatibility',
    'SV4': 'Portability',      'SV5': 'Reliability',     'SV6': 'Efficiency',
    'SV7': 'Energy Preservation', 'SV8': 'Usability',   'SV9': 'Accessibility',
    'SV10': 'Longevity',
}


def load_json_samples(path: Path) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get('samples', [])


def count_values(samples: list[dict]) -> Counter:
    cnt = Counter()
    for s in samples:
        if s.get('has_value_risk'):
            for v in s.get('ground_truth_values', []):
                cnt[v] += 1
    return cnt


# ================================================================
# Load unified datasets
# ================================================================
code_handwritten = load_json_samples(BASE / 'code_scenarios' / 'handwritten.json')
code_pipeline    = load_json_samples(BASE / 'code_scenarios' / 'pipeline.json')
code_generated   = load_json_samples(BASE / 'code_scenarios' / 'generated.json')
text_samples     = load_json_samples(BASE / 'text_scenarios' / 'issues.json')

all_code = code_handwritten + code_pipeline + code_generated

# ================================================================
# Detailed statistics
# ================================================================
print('=' * 60)
print('Dataset Statistics (Unified Format)')
print('=' * 60)

for name, samples in [
    ('Code - Handwritten', code_handwritten),
    ('Code - Pipeline', code_pipeline),
    ('Code - Generated', code_generated),
    ('Text - Issues', text_samples),
]:
    pos = sum(1 for s in samples if s.get('has_value_risk'))
    neg = len(samples) - pos
    val_cnt = count_values(samples)
    print(f'\n[{name}]')
    print(f'  Total:             {len(samples)}')
    print(f'  Has-risk (pos):    {pos} ({pos/len(samples)*100:.1f}%)')
    print(f'  No-risk  (neg):    {neg} ({neg/len(samples)*100:.1f}%)')
    print(f'  Value types:       {len(val_cnt)}')

# Project distribution for text
text_proj = Counter()
for s in text_samples:
    text_proj[s.get('metadata', {}).get('project_name', 'unknown')] += 1

print('\n[Text Scenario - Project distribution]')
for proj, cnt in sorted(text_proj.items()):
    print(f'  {proj:20s}: {cnt:4d}')

# All code combined
code_pos = sum(1 for s in all_code if s.get('has_value_risk'))
code_neg = len(all_code) - code_pos
print(f'\n[All Code Scenarios Combined]')
print(f'  Total:             {len(all_code)} (handwritten={len(code_handwritten)}, pipeline={len(code_pipeline)}, generated={len(code_generated)})')
print(f'  Has-risk (pos):    {code_pos} ({code_pos/len(all_code)*100:.1f}%)')
print(f'  No-risk  (neg):    {code_neg} ({code_neg/len(all_code)*100:.1f}%)')

# ================================================================
# Overall
# ================================================================
total_all = len(all_code) + len(text_samples)
total_pos = code_pos + sum(1 for s in text_samples if s.get('has_value_risk'))
total_neg = total_all - total_pos

combined_val_cnt = count_values(all_code) + count_values(text_samples)
all_val_ids = sorted(combined_val_cnt.keys())

print('\n[Overall]')
print(f'  Total samples:     {total_all}  (code={len(all_code)}, text={len(text_samples)})')
print(f'  Has-risk (pos):    {total_pos} ({total_pos/total_all*100:.1f}%)')
print(f'  No-risk  (neg):    {total_neg} ({total_neg/total_all*100:.1f}%)')
print(f'  Unique value IDs:  {len(all_val_ids)}')
print()
print(f'  {"Label":<6}  {"Value Name":<24}  {"Count":>6}  {"% of pos samples":>16}')
print(f'  {"-"*6}  {"-"*24}  {"-"*6}  {"-"*16}')
for v in all_val_ids:
    cnt = combined_val_cnt[v]
    name = LABEL_NAME.get(v, '?')
    pct = cnt / total_pos * 100
    print(f'  {v:<6}  {name:<24}  {cnt:>6}  {pct:>15.1f}%')

# ================================================================
# Paper-format summary tables
# ================================================================
print('\n')
print('=' * 70)
print('Table (a): Summary statistics of the dataset')
print('=' * 70)
print(f"{'Statistical Indicators':<45} {'Value':<20}")
print('-' * 70)
print(f"{'Total sample size':<45} {total_all}")
print(f"{'Positive instances (with value risks)':<45} {total_pos} ({total_pos/total_all*100:.1f}%)")
print(f"{'Negative instances (without value risks)':<45} {total_neg} ({total_neg/total_all*100:.1f}%)")
print(f"{'Number of value types involved':<45} {len(all_val_ids)}")

print('\n')
print('=' * 70)
print('Table (b): Distribution of human-annotated value labels among positive samples')
print('=' * 70)
print(f"{'ID':<6} {'Value Name':<25} {'Count':<8} {'%':<8}")
print('-' * 70)
for v in all_val_ids:
    cnt = combined_val_cnt[v]
    name = LABEL_NAME.get(v, '?')
    pct = cnt / total_pos * 100
    print(f'{v:<6} {name:<25} {cnt:<8} {pct:.1f}%')
