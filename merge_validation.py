"""
Merge two independent forest edge validation CSVs using the conservative decision rule.

Conservative rule: When disagreement occurs, the lowest-quality assessment is retained.
Quality hierarchy: Y (match) > U (uncertain) > N (no match)
N/A = no image available (missing data, not an assessment)

Merging logic:
- Both N/A → N/A
- N/A + actual assessment → keep the actual assessment
- Both have assessments → keep the lower-quality one (N < U < Y)
"""

import csv
import os

# Quality ranking: lower number = lower quality (more conservative)
QUALITY_RANK = {'N': 0, 'U': 1, 'Y': 2}

def conservative_merge(val_hangkai, val_fujiang):
    """Apply conservative merging rule between two reviewers."""
    h = val_hangkai.strip()
    f = val_fujiang.strip()

    # Both N/A
    if h == 'N/A' and f == 'N/A':
        return 'N/A'
    # One N/A, one has assessment → keep the actual assessment
    if h == 'N/A':
        return f
    if f == 'N/A':
        return h
    # Both have actual assessments → keep the lower quality (conservative)
    h_rank = QUALITY_RANK.get(h, -1)
    f_rank = QUALITY_RANK.get(f, -1)
    if h_rank <= f_rank:
        return h
    else:
        return f

def read_validation_csv(filepath):
    """Read validation CSV, skipping comment/header lines."""
    data = {}
    with open(filepath, 'r', encoding='latin-1') as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            if row[0] == 'Sample_ID':
                header = row
                continue
            if header and row[0].strip().isdigit():
                sample_id = int(row[0].strip())
                data[sample_id] = {
                    'Reviewer_Name': row[1].strip(),
                    'Review_Date': row[2].strip(),
                    'Year_2000': row[3].strip(),
                    'Year_2005': row[4].strip(),
                    'Year_2010': row[5].strip(),
                    'Year_2015': row[6].strip(),
                    'Year_2020': row[7].strip(),
                    'Dynamic': row[8].strip(),
                    'Notes': row[9].strip() if len(row) > 9 else ''
                }
    return data

# File paths
base_dir = r'G:\Hangkai\Global_Forest_edge_mapping_data\Validation_Samples'
hangkai_file = os.path.join(base_dir, 'forest_edge_validation_Hangkai.csv')
fujiang_file = os.path.join(base_dir, 'forest_edge_validation_Fujiang.csv')
merged_file = os.path.join(base_dir, 'forest_edge_validation_merged.csv')

# Read both files
hangkai = read_validation_csv(hangkai_file)
fujiang = read_validation_csv(fujiang_file)

# Verify same sample IDs
assert set(hangkai.keys()) == set(fujiang.keys()), "Sample IDs don't match!"
sample_ids = sorted(hangkai.keys())
print(f"Total samples: {len(sample_ids)}")

# Year columns for static edge consistency
year_cols = ['Year_2000', 'Year_2005', 'Year_2010', 'Year_2015', 'Year_2020']
year_labels = ['2000', '2005', '2010', '2015', '2020']

# ========== INTER-REVIEWER AGREEMENT (before merging) ==========
print("\n" + "="*70)
print("INTER-REVIEWER AGREEMENT (before merging)")
print("="*70)

# For inter-reviewer agreement, compare only where BOTH have actual assessments (Y/N/U)
for col, label in zip(year_cols, year_labels):
    both_assessed = 0
    agree = 0
    for sid in sample_ids:
        h = hangkai[sid][col]
        f = fujiang[sid][col]
        if h in ('Y', 'N', 'U') and f in ('Y', 'N', 'U'):
            both_assessed += 1
            if h == f:
                agree += 1
    if both_assessed > 0:
        pct = agree / both_assessed * 100
        print(f"  {label}: {agree}/{both_assessed} agreement ({pct:.1f}%)")
    else:
        print(f"  {label}: No overlapping assessments")

# Dynamic agreement inter-reviewer
both_dynamic = 0
agree_dynamic = 0
for sid in sample_ids:
    h = hangkai[sid]['Dynamic']
    f = fujiang[sid]['Dynamic']
    if h in ('Y', 'N', 'U') and f in ('Y', 'N', 'U'):
        both_dynamic += 1
        if h == f:
            agree_dynamic += 1
if both_dynamic > 0:
    print(f"  Dynamic: {agree_dynamic}/{both_dynamic} agreement ({agree_dynamic/both_dynamic*100:.1f}%)")

# ========== MERGE ==========
merged = {}
for sid in sample_ids:
    merged[sid] = {}
    for col in year_cols:
        merged[sid][col] = conservative_merge(hangkai[sid][col], fujiang[sid][col])
    merged[sid]['Dynamic'] = conservative_merge(hangkai[sid]['Dynamic'], fujiang[sid]['Dynamic'])
    # Combine notes
    notes_h = hangkai[sid]['Notes']
    notes_f = fujiang[sid]['Notes']
    if notes_h and notes_f:
        merged[sid]['Notes'] = f"[Hangkai] {notes_h}; [Fujiang] {notes_f}"
    elif notes_h:
        merged[sid]['Notes'] = f"[Hangkai] {notes_h}"
    elif notes_f:
        merged[sid]['Notes'] = f"[Fujiang] {notes_f}"
    else:
        merged[sid]['Notes'] = ''

# ========== WRITE MERGED CSV ==========
header_comments = [
    "# Merged Forest Edge Validation Results",
    "#",
    "# This file contains the merged validation results from two independent reviewers:",
    "#   Reviewer 1: Hangkai You (reviewed 2026-02-11)",
    "#   Reviewer 2: Fujiang Ji  (reviewed 2026-02-18)",
    "#",
    "# MERGING METHOD:",
    "# Interpretation was conducted independently by two reviewers to reduce subjective bias.",
    "# Inter-reviewer agreement was evaluated and discrepancies were resolved using a",
    "# conservative decision rule: the lowest-quality assessment was retained whenever",
    "# disagreement occurred.",
    "#",
    "# Quality hierarchy (from lowest to highest): N < U < Y",
    "#   - Y: Predicted edges visually consistent with actual forest-nonforest boundaries",
    "#   - N: Predicted edges do NOT match actual boundaries",
    "#   - U: Unable to determine due to image quality, cloud cover, or ambiguous boundaries",
    "#   - N/A: No image available for this year (treated as missing data)",
    "#",
    "# When one reviewer marked N/A (no imagery) and the other provided an assessment,",
    "# the available assessment was retained.",
    "# When both reviewers had assessments and disagreed, the lower-quality value was kept.",
    "#",
    "# Dynamic_Agreement: Whether observed edge dynamics over time match the product predictions",
    "#   - Y: Product predictions align with observed changes in forest edges over time",
    "#   - N: Product predictions do NOT align with observed changes",
    "#   - U: Unable to determine overall trend",
    "#   - N/A: Insufficient data to assess dynamics",
    "#",
    "# Overall_Notes: Combined notes from both reviewers (prefixed with reviewer name)",
    "#",
]

with open(merged_file, 'w', newline='', encoding='utf-8') as f:
    # Write header comments
    for line in header_comments:
        f.write(line + ',' * 7 + '\n')
    # Write data
    writer = csv.writer(f)
    writer.writerow(['Sample_ID', 'Year_2000_Edge_Consistency', 'Year_2005_Edge_Consistency',
                      'Year_2010_Edge_Consistency', 'Year_2015_Edge_Consistency',
                      'Year_2020_Edge_Consistency', 'Dynamic_Agreement', 'Overall_Notes'])
    for sid in sample_ids:
        writer.writerow([
            sid,
            merged[sid]['Year_2000'],
            merged[sid]['Year_2005'],
            merged[sid]['Year_2010'],
            merged[sid]['Year_2015'],
            merged[sid]['Year_2020'],
            merged[sid]['Dynamic'],
            merged[sid]['Notes']
        ])

print(f"\nMerged CSV written to: {merged_file}")

# ========== MERGED STATISTICS ==========
print("\n" + "="*70)
print("MERGED VALIDATION STATISTICS (Conservative Rule Applied)")
print("="*70)

# Per-year static edge consistency
print("\n--- Static Edge Consistency (per year) ---")
print(f"{'Year':<10} {'Y':>5} {'N':>5} {'U':>5} {'N/A':>5} {'Assessed':>10} {'Match Rate':>12}")
print("-" * 58)

total_Y = 0
total_N = 0
total_U = 0
total_NA = 0

for col, label in zip(year_cols, year_labels):
    counts = {'Y': 0, 'N': 0, 'U': 0, 'N/A': 0}
    for sid in sample_ids:
        val = merged[sid][col]
        counts[val] = counts.get(val, 0) + 1
    assessed = counts['Y'] + counts['N']  # Only Y and N are definitive assessments
    assessed_with_U = counts['Y'] + counts['N'] + counts['U']
    match_rate = counts['Y'] / assessed * 100 if assessed > 0 else 0
    match_rate_with_U = counts['Y'] / assessed_with_U * 100 if assessed_with_U > 0 else 0

    total_Y += counts['Y']
    total_N += counts['N']
    total_U += counts['U']
    total_NA += counts['N/A']

    print(f"{label:<10} {counts['Y']:>5} {counts['N']:>5} {counts['U']:>5} {counts['N/A']:>5} {assessed:>10} {match_rate:>11.1f}%")

# Overall static
total_assessed = total_Y + total_N
total_all = total_Y + total_N + total_U
overall_match = total_Y / total_assessed * 100 if total_assessed > 0 else 0
overall_match_incl_U = total_Y / total_all * 100 if total_all > 0 else 0
print("-" * 58)
print(f"{'Total':<10} {total_Y:>5} {total_N:>5} {total_U:>5} {total_NA:>5} {total_assessed:>10} {overall_match:>11.1f}%")

print(f"\n  Overall agreement (Y out of Y+N):  {total_Y}/{total_assessed} = {overall_match:.1f}%")
print(f"  Overall agreement (Y out of Y+N+U): {total_Y}/{total_all} = {overall_match_incl_U:.1f}%")

# Dynamic consistency
print("\n--- Dynamic Edge Change Agreement ---")
dyn_counts = {'Y': 0, 'N': 0, 'U': 0, 'N/A': 0}
for sid in sample_ids:
    val = merged[sid]['Dynamic']
    dyn_counts[val] = dyn_counts.get(val, 0) + 1

dyn_assessed = dyn_counts['Y'] + dyn_counts['N']
dyn_all = dyn_counts['Y'] + dyn_counts['N'] + dyn_counts['U']
dyn_match = dyn_counts['Y'] / dyn_assessed * 100 if dyn_assessed > 0 else 0

print(f"  Y (agree):     {dyn_counts['Y']}")
print(f"  N (disagree):  {dyn_counts['N']}")
print(f"  U (uncertain): {dyn_counts['U']}")
print(f"  N/A:           {dyn_counts['N/A']}")
print(f"  Agreement (Y out of Y+N): {dyn_counts['Y']}/{dyn_assessed} = {dyn_match:.1f}%")

# Summary table suitable for paper (Table S2 format)
print("\n" + "="*70)
print("SUMMARY TABLE (for Table S2 in paper)")
print("="*70)
print(f"{'Year':<10} {'Comparisons':>12} {'Matches (Y)':>12} {'Non-match (N)':>14} {'Uncertain (U)':>14} {'Match Rate*':>12}")
print("-" * 76)

for col, label in zip(year_cols, year_labels):
    counts = {'Y': 0, 'N': 0, 'U': 0, 'N/A': 0}
    for sid in sample_ids:
        val = merged[sid][col]
        counts[val] = counts.get(val, 0) + 1
    assessed = counts['Y'] + counts['N']
    match_rate = counts['Y'] / assessed * 100 if assessed > 0 else 0
    print(f"{label:<10} {assessed:>12} {counts['Y']:>12} {counts['N']:>14} {counts['U']:>14} {match_rate:>11.1f}%")

print("-" * 76)
print(f"{'Overall':<10} {total_assessed:>12} {total_Y:>12} {total_N:>14} {total_U:>14} {overall_match:>11.1f}%")
print(f"\n* Match Rate = Y / (Y + N), excluding U and N/A from the denominator.")
print(f"  Dynamic agreement: {dyn_counts['Y']}/{dyn_assessed} = {dyn_match:.1f}%")

# Detailed per-sample disagreement log
print("\n" + "="*70)
print("DISAGREEMENT LOG (samples where reviewers differed)")
print("="*70)
for sid in sample_ids:
    for col, label in zip(year_cols + ['Dynamic'], year_labels + ['Dynamic']):
        h = hangkai[sid][col] if col != 'Dynamic' else hangkai[sid]['Dynamic']
        f = fujiang[sid][col] if col != 'Dynamic' else fujiang[sid]['Dynamic']
        m = merged[sid][col] if col != 'Dynamic' else merged[sid]['Dynamic']
        # Only log where both had actual assessments and they disagreed
        if h in ('Y','N','U') and f in ('Y','N','U') and h != f:
            print(f"  Sample {sid:>3}, {label:<10}: Hangkai={h}, Fujiang={f} -> Merged={m}")
