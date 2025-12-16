"""
Regenerate remaining figures - Alignment analysis and mediation paths
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Patch
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Use the same academic color scheme as the first batch (red-blue diverging, similar to examples)
COLORS = {
    'primary': '#2166ac',
    'secondary': '#b2182b',
    'success': '#67a9cf',
    'danger': '#ef8a62',
    'warning': '#fddbc7',
    'info': '#d1e5f0',
    'neutral': '#7f7f7f',
    'light_blue': '#c7e9f1',
    'light_orange': '#fddbc7',
    'light_green': '#f7f7f7',
    'light_red': '#f4a582',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
})

# Load data
df = pd.read_csv('data/analysis/merged_data.csv').dropna()
alignment_bfi = pd.read_csv('data/analysis/alignment_results_bfi.csv')
alignment_liwc = pd.read_csv('data/analysis/alignment_results_liwc.csv')
mediation_summary = pd.read_csv('data/analysis/rq2_mediation_summary.csv')

output_dir = Path('data/analysis/figures')

print("Regenerating remaining figures...")
print("="*60)

# ============================================================
# Figure 6: Mediation Paths (Risk-Taking)
# ============================================================
print("\n6. Mediation Paths - Risk-Taking...")

# Extract risk triplets
risk_mediations = mediation_summary[mediation_summary['Behavior'] == 'Risk Mean Cards']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

triplet_names = [
    ('extraversion', 'Social', 'risk_mean_cards'),
    ('conscientiousness', 'Analytic', 'risk_mean_cards'),
    ('neuroticism', 'emo_neg', 'risk_mean_cards'),
    ('openness', 'tone_pos', 'risk_mean_cards')
]

for idx, (row_idx, row) in enumerate(risk_mediations.iterrows()):
    if idx >= 4:
        break

    ax = axes[idx]

    # Extract values
    a = row['Path_a']
    b = row['Path_b']
    indirect = row['Indirect_effect']
    sig = row['Significant']
    med_type = row['Mediation_type']

    # Parse triplet
    parts = row['Triplet'].split('→')
    X_label = parts[0].title()
    M_label = parts[1]
    Y_label = 'Risk'

    # Node positions
    positions = {'X': (0, 0), 'M': (1, 0.5), 'Y': (2, 0)}

    # Draw nodes
    for node, (x, y) in positions.items():
        circle = Circle((x, y), 0.18, color=COLORS['light_blue'],
                       ec=COLORS['primary'], linewidth=2, zorder=10)
        ax.add_patch(circle)

        label = X_label if node == 'X' else (M_label if node == 'M' else Y_label)
        ax.text(x, y, label, ha='center', va='center',
               fontsize=9, fontweight='bold', zorder=11)

    # Draw paths
    # Path a: X → M
    color_a = COLORS['success'] if abs(a) > 0.1 else COLORS['neutral']
    arrow_a = FancyArrowPatch((0.18, 0.05), (0.82, 0.45),
                             arrowstyle='->', lw=2.5, color=color_a,
                             mutation_scale=20, zorder=5)
    ax.add_patch(arrow_a)
    sig_a = '*' if row['Path_a_p'] < 0.05 else ''
    ax.text(0.5, 0.3, f"a={a:+.2f}{sig_a}",
           ha='center', fontsize=10, color=color_a, fontweight='bold')

    # Path b: M → Y
    color_b = COLORS['success'] if abs(b) > 0.1 else COLORS['neutral']
    arrow_b = FancyArrowPatch((1.18, 0.45), (1.82, 0.05),
                             arrowstyle='->', lw=2.5, color=color_b,
                             mutation_scale=20, zorder=5)
    ax.add_patch(arrow_b)
    sig_b = '*' if row['Path_b_p'] < 0.05 else ''
    ax.text(1.5, 0.3, f"b={b:+.2f}{sig_b}",
           ha='center', fontsize=10, color=color_b, fontweight='bold')

    # Direct effect (dashed)
    arrow_c = FancyArrowPatch((0.18, -0.05), (1.82, -0.05),
                             arrowstyle='->', lw=1.5, linestyle='--',
                             color=COLORS['neutral'], mutation_scale=15, zorder=5)
    ax.add_patch(arrow_c)

    # Indirect effect box
    box_color = COLORS['light_green'] if sig else '#f0f0f0'
    ax.text(1, 0.75, f"Indirect: {indirect:+.3f}{'*' if sig else ''}",
           ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=box_color, edgecolor='black'))

    # Mediation type
    type_color = COLORS['success'] if 'Full' in med_type or 'Partial' in med_type else COLORS['danger']
    ax.text(1, -0.35, med_type, ha='center', fontsize=10,
           fontweight='bold', color=type_color)

    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(-0.6, 1.0)
    ax.axis('off')
    ax.set_title(f'{X_label} → {M_label} → {Y_label}', fontsize=11, fontweight='bold')

plt.suptitle('Mediation Paths: Risk-Taking', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(output_dir / 'mediation_paths_risk_mean_cards.png')
print("   ✓ mediation_paths_risk_mean_cards.png")
plt.close()

# ============================================================
# Figure 7: Mediation Paths (Sycophancy)
# ============================================================
print("\n7. Mediation Paths - Sycophancy...")

syc_mediations = mediation_summary[mediation_summary['Behavior'] == 'Sycophancy Rate']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (row_idx, row) in enumerate(syc_mediations.iterrows()):
    if idx >= 4:
        break

    ax = axes[idx]

    # Extract values
    a = row['Path_a']
    b = row['Path_b']
    indirect = row['Indirect_effect']
    sig = row['Significant']
    med_type = row['Mediation_type']

    parts = row['Triplet'].split('→')
    X_label = parts[0].title()
    M_label = parts[1]
    Y_label = 'Sycophancy'

    positions = {'X': (0, 0), 'M': (1, 0.5), 'Y': (2, 0)}

    for node, (x, y) in positions.items():
        circle = Circle((x, y), 0.18, color=COLORS['light_orange'],
                       ec=COLORS['secondary'], linewidth=2, zorder=10)
        ax.add_patch(circle)

        label = X_label if node == 'X' else (M_label if node == 'M' else Y_label)
        ax.text(x, y, label, ha='center', va='center',
               fontsize=9, fontweight='bold', zorder=11)

    # Paths
    color_a = COLORS['success'] if abs(a) > 0.1 else COLORS['neutral']
    arrow_a = FancyArrowPatch((0.18, 0.05), (0.82, 0.45),
                             arrowstyle='->', lw=2.5, color=color_a,
                             mutation_scale=20, zorder=5)
    ax.add_patch(arrow_a)
    sig_a = '*' if row['Path_a_p'] < 0.05 else ''
    ax.text(0.5, 0.3, f"a={a:+.2f}{sig_a}",
           ha='center', fontsize=10, color=color_a, fontweight='bold')

    color_b = COLORS['success'] if abs(b) > 0.1 else COLORS['neutral']
    arrow_b = FancyArrowPatch((1.18, 0.45), (1.82, 0.05),
                             arrowstyle='->', lw=2.5, color=color_b,
                             mutation_scale=20, zorder=5)
    ax.add_patch(arrow_b)
    sig_b = '*' if row['Path_b_p'] < 0.05 else ''
    ax.text(1.5, 0.3, f"b={b:+.2f}{sig_b}",
           ha='center', fontsize=10, color=color_b, fontweight='bold')

    arrow_c = FancyArrowPatch((0.18, -0.05), (1.82, -0.05),
                             arrowstyle='->', lw=1.5, linestyle='--',
                             color=COLORS['neutral'], mutation_scale=15, zorder=5)
    ax.add_patch(arrow_c)

    box_color = COLORS['light_green'] if sig else '#f0f0f0'
    ax.text(1, 0.75, f"Indirect: {indirect:+.3f}{'*' if sig else ''}",
           ha='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=box_color, edgecolor='black'))

    type_color = COLORS['success'] if 'Full' in med_type or 'Partial' in med_type else COLORS['danger']
    ax.text(1, -0.35, med_type, ha='center', fontsize=10,
           fontweight='bold', color=type_color)

    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(-0.6, 1.0)
    ax.axis('off')
    ax.set_title(f'{X_label} → {M_label} → {Y_label}', fontsize=11, fontweight='bold')

plt.suptitle('Mediation Paths: Sycophancy', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(output_dir / 'mediation_paths_sycophancy_rate.png')
print("   ✓ mediation_paths_sycophancy_rate.png")
plt.close()

# ============================================================
# Figure 8: Alignment Percentages
# ============================================================
print("\n8. Alignment Percentages...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# BFI
ax = axes[0]
bfi_summary = []
for behavior in ['risk_mean_cards', 'sycophancy_rate']:
    df_b = alignment_bfi[alignment_bfi['behavior'] == behavior]
    total = len(df_b)
    aligned = df_b['aligned'].sum()
    pct = aligned / total * 100 if total > 0 else 0

    df_directional = df_b[df_b['expected_direction'] != 0]
    if len(df_directional) > 0:
        dir_aligned = df_directional['aligned'].sum()
        dir_pct = dir_aligned / len(df_directional) * 100
    else:
        dir_pct = 0

    label = 'Risk-Taking' if 'risk' in behavior else 'Sycophancy'
    bfi_summary.append({'behavior': label, 'overall': pct, 'directional': dir_pct})

df_bfi_sum = pd.DataFrame(bfi_summary)
x = np.arange(len(df_bfi_sum))
width = 0.35

bars1 = ax.bar(x - width/2, df_bfi_sum['overall'], width, label='Overall',
              alpha=0.9, color=COLORS['primary'], edgecolor='black')
bars2 = ax.bar(x + width/2, df_bfi_sum['directional'], width, label='Directional Only',
              alpha=0.9, color=COLORS['secondary'], edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(50, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.7,
          label='Chance (50%)')
ax.set_ylabel('Alignment Percentage (%)', fontweight='bold')
ax.set_title('BFI Trait-Behavior Alignment', fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(df_bfi_sum['behavior'], fontweight='bold')
ax.legend(frameon=True, fancybox=True, shadow=True)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# LIWC
ax = axes[1]
liwc_summary = []
for behavior in ['risk_mean_cards', 'sycophancy_rate']:
    df_b = alignment_liwc[alignment_liwc['behavior'] == behavior]
    total = len(df_b)
    aligned = df_b['aligned'].sum()
    pct = aligned / total * 100 if total > 0 else 0

    df_directional = df_b[df_b['expected_direction'] != 0]
    if len(df_directional) > 0:
        dir_aligned = df_directional['aligned'].sum()
        dir_pct = dir_aligned / len(df_directional) * 100
    else:
        dir_pct = 0

    label = 'Risk-Taking' if 'risk' in behavior else 'Sycophancy'
    liwc_summary.append({'behavior': label, 'overall': pct, 'directional': dir_pct})

df_liwc_sum = pd.DataFrame(liwc_summary)

bars1 = ax.bar(x - width/2, df_liwc_sum['overall'], width, label='Overall',
              alpha=0.9, color=COLORS['primary'], edgecolor='black')
bars2 = ax.bar(x + width/2, df_liwc_sum['directional'], width, label='Directional Only',
              alpha=0.9, color=COLORS['secondary'], edgecolor='black')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.axhline(50, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.7,
          label='Chance (50%)')
ax.set_ylabel('Alignment Percentage (%)', fontweight='bold')
ax.set_title('LIWC Feature-Behavior Alignment', fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(df_liwc_sum['behavior'], fontweight='bold')
ax.legend(frameon=True, fancybox=True, shadow=True)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / 'alignment_percentages.png')
print("   ✓ alignment_percentages.png")
plt.close()

print("\nContinuing with heatmaps and other figures...")
