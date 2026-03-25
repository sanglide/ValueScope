import json
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent / 'data'

print('=' * 60)
print('Dataset Statistics')
print('=' * 60)

# ================================================================
# Code Scenario
# ================================================================
with open(BASE / 'sample_scenarios.json', encoding='utf-8') as f:
    code_data = json.load(f)
code_samples = code_data if isinstance(code_data, list) else code_data.get('samples', [])
code_pos = sum(1 for s in code_samples if s.get('has_value_risk', False))
code_neg = len(code_samples) - code_pos
code_val_cnt = Counter()
for s in code_samples:
    for v in s.get('ground_truth_values', []):
        code_val_cnt[v] += 1

print('\n[Code Scenario - sample_scenarios.json]')
print(f'  Total:             {len(code_samples)}')
print(f'  Has-risk (pos):    {code_pos} ({code_pos/len(code_samples)*100:.1f}%)')
print(f'  No-risk  (neg):    {code_neg} ({code_neg/len(code_samples)*100:.1f}%)')
print(f'  Value types:       {len(code_val_cnt)}')
print(f'  Value distribution:{dict(sorted(code_val_cnt.items()))}')

# ================================================================
# Text Scenario (values-issues-dataset-master)
# ================================================================
issues_dir = BASE / 'values-issues-dataset-master'

# --- issues.csv (pipe-separated) ---
issues_raw = (issues_dir / 'issues.csv').read_text(encoding='utf-8').strip().split('\n')
issue_rows = []
for line in issues_raw[1:]:
    parts = line.split('|')
    if len(parts) >= 3:
        issue_rows.append({
            'issue_id':     parts[0].strip().strip('"'),  # strip surrounding quotes
            'project_name': parts[2].strip().strip('"'),
        })

# --- proposed_values_id -> HV/SV label (from data_loader.py ISSUES_VALUE_ID_MAPPING) ---
# proposed-values.csv: id|value_name
# values-label.csv: issue_id|post_id|proposed_values_id
MAPPING = {
    '1':  None,     # None -> no value risk
    '2':  'HV9',    # Privacy
    '3':  'HV6',    # Freedom
    '4':  'HV7',    # Independence
    '5':  'SV6',    # Efficiency
    '6':  'HV10',   # Security
    '7':  'SV10',   # Longevity
    '10': 'SV8',    # Usability
    '11': 'SV9',    # Accessibility
    '12': 'HV2',    # Pleasure
    '13': 'SV2',    # Correctness
    '14': 'HV5',    # Sense of Belonging
    '15': 'HV1',    # Conformity
    '16': 'SV1',    # Trust
    '17': 'HV3',    # Dignity
    '18': 'HV8',    # Wealth
    '19': 'HV4',    # Inclusiveness
    '20': 'SV5',    # Reliability
    '21': 'SV3',    # Compatibility
    '22': 'SV4',    # Portability
    '23': 'SV7',    # Energy Preservation
}

# Value label -> English name (for display)
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

labels_raw = (issues_dir / 'values-label.csv').read_text(encoding='utf-8').strip().split('\n')
issue_values = {}
for line in labels_raw[1:]:
    parts = line.split('|')
    if len(parts) >= 3:
        iid = parts[0].strip()
        vid = parts[2].strip()
        mapped = MAPPING.get(vid)
        if mapped:
            issue_values.setdefault(iid, set()).add(mapped)

# --- posts: pipe-separated, col[1] = issue_id ---
posts_raw = (issues_dir / 'issue-posts.csv').read_text(encoding='utf-8').strip().split('\n')
posts_per_issue = Counter()
total_posts = 0
for line in posts_raw[1:]:
    parts = line.split('|')
    if len(parts) >= 2:
        iid = parts[1].strip().strip('"')
        posts_per_issue[iid] += 1
        total_posts += 1
avg_posts = total_posts / len(posts_per_issue) if posts_per_issue else 0

# --- aggregate by project ---
proj_stats = {}
all_val_cnt = Counter()
for row in issue_rows:
    iid, proj = row['issue_id'], row['project_name']
    proj_stats.setdefault(proj, {'total': 0, 'pos': 0, 'neg': 0})
    gt = sorted(issue_values.get(iid, set()))
    has_risk = len(gt) > 0
    proj_stats[proj]['total'] += 1
    if has_risk:
        proj_stats[proj]['pos'] += 1
        for v in gt:
            all_val_cnt[v] += 1
    else:
        proj_stats[proj]['neg'] += 1

total_text = sum(s['total'] for s in proj_stats.values())
total_pos  = sum(s['pos']   for s in proj_stats.values())
total_neg  = sum(s['neg']   for s in proj_stats.values())

print('\n[Text Scenario - values-issues-dataset-master]')
print(f'  Total issues:      {total_text}')
print(f'  Has-risk (pos):    {total_pos} ({total_pos/total_text*100:.1f}%)')
print(f'  No-risk  (neg):    {total_neg} ({total_neg/total_text*100:.1f}%)')
print(f'  Value types:       {len(all_val_cnt)}')
print(f'  Total posts:       {total_posts}')
print(f'  Avg posts/issue:   {avg_posts:.1f}')
print()
for proj, st in sorted(proj_stats.items()):
    pct_p = st['pos'] / st['total'] * 100
    pct_n = st['neg'] / st['total'] * 100
    print(f'  {proj:20s}: total={st["total"]:4d}  pos={st["pos"]:4d} ({pct_p:.1f}%)  neg={st["neg"]:4d} ({pct_n:.1f}%)')

print()
print('  Value label distribution (in positive samples):')
for v, cnt in sorted(all_val_cnt.items()):
    name = LABEL_NAME.get(v, '?')
    print(f'    {v:5s} ({name:22s}): {cnt:4d}')

# ================================================================
# Overall
# ================================================================
total_all = len(code_samples) + total_text
total_all_pos = code_pos + total_pos
total_all_neg = code_neg + total_neg

# Merge code + text value labels
combined_val_cnt = Counter(all_val_cnt)
for v, c in code_val_cnt.items():
    combined_val_cnt[v] += c

all_val_ids = sorted(combined_val_cnt.keys())

print('\n[Overall]')
print(f'  Total samples:     {total_all}  (code={len(code_samples)}, text={total_text})')
print(f'  Has-risk (pos):    {total_all_pos} ({total_all_pos/total_all*100:.1f}%)')
print(f'  No-risk  (neg):    {total_all_neg} ({total_all_neg/total_all*100:.1f}%)')
print(f'  Unique value IDs:  {len(all_val_ids)}')
print()
print(f'  {"Label":<6}  {"Value Name":<24}  {"Count":>6}  {"% of pos samples":>16}')
print(f'  {"-"*6}  {"-"*24}  {"-"*6}  {"-"*16}')
for v in sorted(combined_val_cnt.keys()):
    cnt = combined_val_cnt[v]
    name = LABEL_NAME.get(v, '?')
    pct = cnt / total_all_pos * 100
    print(f'  {v:<6}  {name:<24}  {cnt:>6}  {pct:>15.1f}%')
