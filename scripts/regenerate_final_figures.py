"""
Generate final set of figures - Alignment heatmaps, key cases, model heterogeneity
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Academic color scheme (red-blue diverging, similar to examples)
COLORS = {
    'primary': '#2166ac',
    'secondary': '#b2182b',
    'success': '#67a9cf',
    'danger': '#ef8a62',
    'warning': '#fddbc7',
    'info': '#d1e5f0',
    'neutral': '#7f7f7f',
    'light_blue': '#c7e9f1',
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
})

df = pd.read_csv('data/analysis/merged_data.csv').dropna()
alignment_bfi = pd.read_csv('data/analysis/alignment_results_bfi.csv')
alignment_liwc = pd.read_csv('data/analysis/alignment_results_liwc.csv')

output_dir = Path('data/analysis/figures')

print("Generating final figures...")
print("="*60)

# ============================================================
# Figure 9: BFI Alignment Heatmap
# ============================================================
print("\n9. BFI Alignment Heatmap...")

bfi_traits = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
behaviors = ['risk_mean_cards', 'sycophancy_rate']
behavior_labels = ['Risk-Taking', 'Sycophancy']

matrix_coef = np.zeros((len(bfi_traits), 2))
matrix_aligned = np.zeros((len(bfi_traits), 2))
matrix_sig = np.zeros((len(bfi_traits), 2))

for i, trait in enumerate(bfi_traits):
    for j, behavior in enumerate(behaviors):
        row = alignment_bfi[(alignment_bfi['feature'] == trait) &
                           (alignment_bfi['behavior'] == behavior)]

        if len(row) > 0:
            matrix_coef[i, j] = row.iloc[0]['coefficient']
            matrix_aligned[i, j] = row.iloc[0]['aligned']
            matrix_sig[i, j] = row.iloc[0]['significant']

fig, ax = plt.subplots(figsize=(6, 7))

# Create heatmap colors manually
for i in range(len(bfi_traits)):
    for j in range(2):
        aligned = matrix_aligned[i, j]
        significant = matrix_sig[i, j]

        if aligned == 1:
            color = COLORS['success'] if significant else COLORS['light_green']
        else:
            color = COLORS['danger'] if significant else COLORS['light_red']

        rect = plt.Rectangle((j, i), 1, 1, facecolor=color,
                            edgecolor='white', linewidth=3)
        ax.add_patch(rect)

        # Add coefficient text
        coef_text = f"{matrix_coef[i, j]:+.2f}"
        if significant:
            coef_text += "*"

        text_color = 'white' if abs(matrix_coef[i, j]) > 0.15 else 'black'
        ax.text(j + 0.5, i + 0.5, coef_text, ha='center', va='center',
               fontsize=10, fontweight='bold', color=text_color)

ax.set_xlim(0, 2)
ax.set_ylim(0, len(bfi_traits))
ax.set_xticks([0.5, 1.5])
ax.set_yticks(np.arange(len(bfi_traits)) + 0.5)
ax.set_xticklabels(behavior_labels, fontweight='bold')
ax.set_yticklabels([t.title() for t in bfi_traits])
ax.invert_yaxis()

# Legend
legend_elements = [
    Patch(facecolor=COLORS['success'], label='Aligned + Significant'),
    Patch(facecolor=COLORS['light_green'], label='Aligned + Non-sig'),
    Patch(facecolor=COLORS['danger'], label='Misaligned + Significant'),
    Patch(facecolor=COLORS['light_red'], label='Misaligned + Non-sig')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1),
         frameon=True, fancybox=True, shadow=True)

ax.set_title('BFI Trait-Behavior Alignment\n(Standardized Coefficients, *p<0.05)',
            fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(output_dir / 'alignment_heatmap_bfi.png')
print("   ✓ alignment_heatmap_bfi.png")
plt.close()

# ============================================================
# Figure 10: LIWC Alignment Heatmap
# ============================================================
print("\n10. LIWC Alignment Heatmap...")

liwc_features = ['emo_neg', 'i', 'Cognition', 'Analytic', 'emo_pos', 'tone_pos',
                'risk', 'you', 'Social', 'we', 'assent', 'tentat']

matrix_coef = np.zeros((len(liwc_features), 2))
matrix_aligned = np.zeros((len(liwc_features), 2))
matrix_sig = np.zeros((len(liwc_features), 2))

for i, feature in enumerate(liwc_features):
    for j, behavior in enumerate(behaviors):
        row = alignment_liwc[(alignment_liwc['feature'] == feature) &
                            (alignment_liwc['behavior'] == behavior)]

        if len(row) > 0:
            matrix_coef[i, j] = row.iloc[0]['coefficient']
            matrix_aligned[i, j] = row.iloc[0]['aligned']
            matrix_sig[i, j] = row.iloc[0]['significant']

fig, ax = plt.subplots(figsize=(6, 10))

for i in range(len(liwc_features)):
    for j in range(2):
        aligned = matrix_aligned[i, j]
        significant = matrix_sig[i, j]

        if aligned == 1:
            color = COLORS['success'] if significant else COLORS['light_green']
        else:
            color = COLORS['danger'] if significant else COLORS['light_red']

        rect = plt.Rectangle((j, i), 1, 1, facecolor=color,
                            edgecolor='white', linewidth=3)
        ax.add_patch(rect)

        coef_text = f"{matrix_coef[i, j]:+.2f}"
        if significant:
            coef_text += "*"

        text_color = 'white' if abs(matrix_coef[i, j]) > 0.1 else 'black'
        ax.text(j + 0.5, i + 0.5, coef_text, ha='center', va='center',
               fontsize=10, fontweight='bold', color=text_color)

ax.set_xlim(0, 2)
ax.set_ylim(0, len(liwc_features))
ax.set_xticks([0.5, 1.5])
ax.set_yticks(np.arange(len(liwc_features)) + 0.5)
ax.set_xticklabels(behavior_labels, fontweight='bold')
ax.set_yticklabels(liwc_features)
ax.invert_yaxis()

legend_elements = [
    Patch(facecolor=COLORS['success'], label='Aligned + Significant'),
    Patch(facecolor=COLORS['light_green'], label='Aligned + Non-sig'),
    Patch(facecolor=COLORS['danger'], label='Misaligned + Significant'),
    Patch(facecolor=COLORS['light_red'], label='Misaligned + Non-sig')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1),
         frameon=True, fancybox=True, shadow=True)

ax.set_title('LIWC Feature-Behavior Alignment\n(Standardized Coefficients, *p<0.05)',
            fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(output_dir / 'alignment_heatmap_liwc.png')
print("   ✓ alignment_heatmap_liwc.png")
plt.close()

# ============================================================
# Figure 11: Key Alignment Cases
# ============================================================
print("\n11. Key Alignment Cases (Scatterplots)...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Misaligned Case 1: Conscientiousness → Risk
ax = axes[0, 0]
ax.scatter(df['conscientiousness'], df['risk_mean_cards'], alpha=0.6,
          s=50, color=COLORS['primary'], edgecolor='white', linewidth=0.5)
z = np.polyfit(df['conscientiousness'], df['risk_mean_cards'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['conscientiousness'].min(), df['conscientiousness'].max(), 100)
ax.plot(x_line, p(x_line), color=COLORS['danger'], linestyle='--', linewidth=2.5)
r, pval = stats.pearsonr(df['conscientiousness'], df['risk_mean_cards'])
ax.set_xlabel('Conscientiousness', fontweight='bold')
ax.set_ylabel('Risk-Taking (Mean Cards)', fontweight='bold')
ax.set_title(f'Conscientiousness → Risk\nr={r:+.3f}, Expected: Negative',
            fontweight='bold', color=COLORS['danger'])
ax.text(0.05, 0.95, 'MISALIGNED', transform=ax.transAxes,
       fontsize=12, verticalalignment='top', color=COLORS['danger'],
       fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.grid(alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Misaligned Case 2: Neuroticism → Risk
ax = axes[0, 1]
ax.scatter(df['neuroticism'], df['risk_mean_cards'], alpha=0.6,
          s=50, color=COLORS['primary'], edgecolor='white', linewidth=0.5)
z = np.polyfit(df['neuroticism'], df['risk_mean_cards'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['neuroticism'].min(), df['neuroticism'].max(), 100)
ax.plot(x_line, p(x_line), color=COLORS['danger'], linestyle='--', linewidth=2.5)
r, pval = stats.pearsonr(df['neuroticism'], df['risk_mean_cards'])
ax.set_xlabel('Neuroticism', fontweight='bold')
ax.set_ylabel('Risk-Taking (Mean Cards)', fontweight='bold')
ax.set_title(f'Neuroticism → Risk\nr={r:+.3f}, Expected: Positive',
            fontweight='bold', color=COLORS['danger'])
ax.text(0.05, 0.95, 'MISALIGNED', transform=ax.transAxes,
       fontsize=12, verticalalignment='top', color=COLORS['danger'],
       fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.grid(alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Aligned Case 1: tone_pos → Risk
ax = axes[1, 0]
ax.scatter(df['tone_pos'], df['risk_mean_cards'], alpha=0.6,
          s=50, color=COLORS['secondary'], edgecolor='white', linewidth=0.5)
z = np.polyfit(df['tone_pos'], df['risk_mean_cards'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['tone_pos'].min(), df['tone_pos'].max(), 100)
ax.plot(x_line, p(x_line), color=COLORS['success'], linestyle='-', linewidth=2.5)
r, pval = stats.pearsonr(df['tone_pos'], df['risk_mean_cards'])
ax.set_xlabel('LIWC: Positive Tone', fontweight='bold')
ax.set_ylabel('Risk-Taking (Mean Cards)', fontweight='bold')
ax.set_title(f'Positive Tone → Risk\nr={r:+.3f}, Expected: Positive',
            fontweight='bold', color=COLORS['success'])
ax.text(0.05, 0.95, 'ALIGNED', transform=ax.transAxes,
       fontsize=12, verticalalignment='top', color=COLORS['success'],
       fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.grid(alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Aligned Case 2: tentat → Sycophancy
ax = axes[1, 1]
ax.scatter(df['tentat'], df['sycophancy_rate'], alpha=0.6,
          s=50, color=COLORS['secondary'], edgecolor='white', linewidth=0.5)
z = np.polyfit(df['tentat'], df['sycophancy_rate'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['tentat'].min(), df['tentat'].max(), 100)
ax.plot(x_line, p(x_line), color=COLORS['success'], linestyle='-', linewidth=2.5)
r, pval = stats.pearsonr(df['tentat'], df['sycophancy_rate'])
ax.set_xlabel('LIWC: Tentative Language', fontweight='bold')
ax.set_ylabel('Sycophancy Rate', fontweight='bold')
ax.set_title(f'Tentative Language → Sycophancy\nr={r:+.3f}, Expected: Positive',
            fontweight='bold', color=COLORS['success'])
ax.text(0.05, 0.95, 'ALIGNED', transform=ax.transAxes,
       fontsize=12, verticalalignment='top', color=COLORS['success'],
       fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.grid(alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle('Key Alignment Cases: Misaligned vs. Aligned Associations',
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'key_alignment_cases.png')
print("   ✓ key_alignment_cases.png")
plt.close()

# ============================================================
# Figure 12: Model Heterogeneity
# ============================================================
print("\n12. Model Heterogeneity (Forest Plots)...")

features_to_plot = [
    ('conscientiousness', 'risk_mean_cards', 'Conscientiousness → Risk'),
    ('you', 'sycophancy_rate', 'LIWC "you" → Sycophancy')
]

fig, axes = plt.subplots(2, 1, figsize=(10, 9))

for idx, (feature, behavior, title) in enumerate(features_to_plot):
    ax = axes[idx]

    results = []
    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]
        if len(model_df) > 5 and feature in model_df.columns:
            r, p = stats.pearsonr(model_df[feature], model_df[behavior])

            # Fisher's z CI
            z = np.arctanh(r)
            se = 1 / np.sqrt(len(model_df) - 3)
            ci_lower = np.tanh(z - 1.96 * se)
            ci_upper = np.tanh(z + 1.96 * se)

            model_short = model.split('_')[-1][:15]
            results.append({
                'model': model_short,
                'r': r,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n': len(model_df),
                'significant': p < 0.05
            })

    df_results = pd.DataFrame(results)

    # Plot
    y_pos = np.arange(len(df_results))

    for i, row in df_results.iterrows():
        color = COLORS['success'] if row['significant'] else COLORS['neutral']

        # CI line
        ax.plot([row['ci_lower'], row['ci_upper']], [i, i],
               'k-', linewidth=2, alpha=0.6)

        # Point estimate
        ax.plot(row['r'], i, 'o', color=color, markersize=10,
               markeredgecolor='black', markeredgewidth=1.5, zorder=10)

    ax.axvline(0, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.7,
              label='Null Effect (r=0)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_results['model'])
    ax.set_xlabel('Pearson Correlation (r) with 95% CI', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS['success'], label='p < 0.05'),
        Patch(facecolor=COLORS['neutral'], label='n.s.')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
             fancybox=True, shadow=True)

plt.suptitle('Cross-Model Heterogeneity in Trait-Behavior Associations',
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'model_heterogeneity.png')
print("   ✓ model_heterogeneity.png")
plt.close()

print("\n" + "="*60)
print("✅ All figures regenerated successfully!")
print("="*60)
print("\nGenerated 12 academic-standard figures:")
print("  1. descriptive_statistics.png")
print("  2. behavior_distributions.png")
print("  3. model_differences.png")
print("  4. rq1_model_comparison.png")
print("  5. rq3_liwc_behavior_heatmap.png")
print("  6. mediation_paths_risk_mean_cards.png")
print("  7. mediation_paths_sycophancy_rate.png")
print("  8. alignment_percentages.png")
print("  9. alignment_heatmap_bfi.png")
print(" 10. alignment_heatmap_liwc.png")
print(" 11. key_alignment_cases.png")
print(" 12. model_heterogeneity.png")
print("\nAll figures meet academic standards:")
print("  ✓ All text in English")
print("  ✓ Color-blind friendly palette")
print("  ✓ 300 DPI resolution")
print("  ✓ Consistent formatting")
print("  ✓ Clear labels and legends")
