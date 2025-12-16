#!/usr/bin/env python3
"""
Replace pandoc-generated figure and table code with standard ACL environments
"""

import re

# Read the tex file
with open('essay/essay_en_0.1_final.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Define figure replacements (filename -> label)
figures = {
    'behavior_distributions.png': ('fig:behavior_dist', 'Distributions of behavioral outcomes. Left: risk decision-making (CCT number of cards); right: sycophancy (conformity rate). The red dashed line indicates the sample mean.'),
    'model_differences.png': ('fig:model_diff', 'Boxplots of behavioral distributions across models. Results show pronounced differences in behavioral baselines between models.'),
    'rq1_model_comparison.png': ('fig:rq1', r'Comparison of predictive validity (RQ1). Bar height represents adjusted $R^2$. Error bars denote bootstrap standard errors. Asterisks indicate significant improvement over the null model.'),
    'mediation_paths_risk_mean_cards.png': ('fig:mediation_risk', 'Mediation paths for risk decision-making. Green paths indicate significant effects. The Extraversion path shows full mediation, whereas the Neuroticism path exhibits a suppression effect.', True),  # wide figure
    'mediation_paths_sycophancy_rate.png': ('fig:mediation_syco', 'Mediation paths for sycophancy. Note the break at Path b for Agreeableness, indicating that language style fails to translate into actual sycophantic behavior.', True),  # wide figure
    'alignment_percentages.png': ('fig:alignment', r'Theoretical alignment percentages for feature--behavior associations. The red line represents the 50\% random level. Results indicate systematic deviations between LLM behavioral mechanisms and human psychology.'),
    'rq3_liwc_behavior_heatmap.png': ('fig:heatmap', 'Heatmap of correlations between LIWC features and behavioral outcomes. Color represents the magnitude of correlation coefficients (red negative, blue positive).'),
}

# Pattern to match pandocbounded images and their captions
pattern = r'\\pandocbounded\{\\includegraphics\[keepaspectratio,alt=\{[^}]*\}\]\{figures/([^}]+)\}\}\s*\\emph\{Figure \d+\.\s*([^}]+)\}'

def replace_figure(match):
    filename = match.group(1)

    if filename not in figures:
        print(f"Warning: {filename} not in predefined figures")
        return match.group(0)

    fig_info = figures[filename]
    label = fig_info[0]
    caption = fig_info[1]
    is_wide = len(fig_info) > 2 and fig_info[2]

    if is_wide:
        # Two-column figure
        return f'''\\begin{{figure*}}[t]
\\centering
\\includegraphics[width=0.8\\textwidth]{{figures/{filename}}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure*}}'''
    else:
        # Single-column figure
        return f'''\\begin{{figure}}[t]
\\centering
\\includegraphics[width=\\columnwidth]{{figures/{filename}}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}'''

# Replace all figures
content_new = re.sub(pattern, replace_figure, content, flags=re.DOTALL | re.MULTILINE)

# ===== TABLE REPLACEMENT =====

# Standard ACL table to replace the longtable
table_replacement = r'''\\begin{table*}[t]
\\centering
\\caption{Descriptive statistics for BFI traits, behavioral outcomes, and key LIWC features (N = 249).}
\\label{tab:descriptive}
\\begin{tabular}{lrrrr}
\\toprule
\\textbf{Variable} & \\textbf{Mean} & \\textbf{SD} & \\textbf{Min} & \\textbf{Max} \\\\
\\midrule
\\multicolumn{5}{l}{\\textit{BFI Personality Traits}} \\\\
\\quad Openness            & 3.39  & 1.08  & 1.40  & 4.80  \\\\
\\quad Conscientiousness   & 3.87  & 0.93  & 1.44  & 5.00  \\\\
\\quad Extraversion        & 3.33  & 1.30  & 1.00  & 5.00  \\\\
\\quad Agreeableness       & 3.58  & 0.96  & 1.22  & 5.00  \\\\
\\quad Neuroticism         & 2.83  & 1.27  & 1.00  & 5.00  \\\\
\\midrule
\\multicolumn{5}{l}{\\textit{Behavioral Outcomes}} \\\\
\\quad Risk-Taking (Mean Cards)  & 21.84 & 6.19  & 2.96  & 29.52 \\\\
\\quad Sycophancy Rate           & 0.10  & 0.14  & 0.00  & 0.62  \\\\
\\midrule
\\multicolumn{5}{l}{\\textit{LIWC Linguistic Features}} \\\\
\\quad Negative Emotion   & 0.62  & 0.47  & 0.00  & 1.93  \\\\
\\quad I (First Person)   & 8.91  & 3.21  & 0.00  & 16.34 \\\\
\\quad Cognition          & 22.71 & 3.00  & 14.62 & 29.95 \\\\
\\quad Analytic           & 67.80 & 19.49 & 16.81 & 98.02 \\\\
\\quad Positive Emotion   & 1.60  & 0.71  & 0.13  & 3.91  \\\\
\\quad Positive Tone      & 4.54  & 1.34  & 0.99  & 8.49  \\\\
\\quad You (Second Person) & 0.22 & 0.39  & 0.00  & 1.98  \\\\
\\quad Social             & 8.52  & 3.65  & 1.71  & 19.03 \\\\
\\quad We (First Person Plural) & 0.90 & 1.14 & 0.00 & 6.62  \\\\
\\quad Tentative          & 1.57  & 0.82  & 0.11  & 4.43  \\\\
\\bottomrule
\\end{tabular}
\\end{table*}'''

# Pattern to match the longtable environment and its caption
# This matches from {\def\LTcaptype to the \emph{Table 1. ...} after \end{longtable}
table_pattern = r'\{\\def\\LTcaptype\{none\}.*?\\begin\{longtable\}.*?\\end\{longtable\}\s*\}\s*\\emph\{Table 1\..*?\}'

# Replace the table
content_new = re.sub(table_pattern, table_replacement, content_new, flags=re.DOTALL)

# Write the result
with open('essay/essay_en_0.1_fixed.tex', 'w', encoding='utf-8') as f:
    f.write(content_new)

print("✅ Replacement complete!")
print("📄 Output: essay/essay_en_0.1_fixed.tex")
print("\n📊 Replaced figures:")
for i, (filename, info) in enumerate(figures.items(), 1):
    print(f"  {i}. {filename} -> {info[0]}")
print("\n📋 Replaced table:")
print("  ✓ Table 1: Descriptive statistics (longtable -> table*)")
