"""
RQ2: Mediation Analysis
Test the triangular verification framework: BFI → LIWC → Behavior

Theoretical framework:
1. Jiang et al.: BFI ≈ LIWC (path a)
2. Han et al.: BFI ≠ Behavior (path c weak)
3. This study: LIWC → Behavior (path b, RQ1)
4. RQ2 core: BFI → LIWC → Behavior indirect effect (a×b)

Analysis methods:
- Baron & Kenny four-step method
- Bootstrap indirect effect 95% CI (Preacher & Hayes)
- Model-level analysis (control model heterogeneity)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings('ignore')


class MediationAnalysis:
    def __init__(self, data_path='data/analysis/merged_data.csv'):
        """初始化"""
        self.df = pd.read_csv(data_path).dropna()
        self.output_dir = Path('data/analysis')
        self.output_dir.mkdir(exist_ok=True)
        self.fig_dir = self.output_dir / 'figures'
        self.fig_dir.mkdir(exist_ok=True)

        print(f"Data loaded: {len(self.df)} rows")

        # Define variables
        self.bfi_traits = ['openness', 'conscientiousness', 'extraversion',
                          'agreeableness', 'neuroticism']
        self.behaviors = ['risk_mean_cards', 'sycophancy_rate']

        # Key mediation triplets (based on theoretical expectations and RQ1 findings)
        self.mediations = {
            'risk_mean_cards': [
                # (BFI trait, LIWC mediator, theoretical expectation)
                ('extraversion', 'Social', 'positive'),  # Extraversion → Social → Risk
                ('conscientiousness', 'Analytic', 'negative'),  # Conscientiousness → Analytic → Risk
                ('neuroticism', 'emo_neg', 'complex'),  # Neuroticism → emo_neg → Complex
                ('openness', 'tone_pos', 'positive'),  # Openness → tone_pos → Exploration
            ],
            'sycophancy_rate': [
                ('agreeableness', 'Social', 'positive'),  # Agreeableness → Social → Compliance
                ('openness', 'tentat', 'negative'),  # Openness → tentat → Independence
                ('neuroticism', 'emo_neg', 'positive'),  # Neuroticism → emo_neg → Anxiety Compliance
                ('conscientiousness', 'we', 'positive'),  # Conscientiousness → we → Normative
            ]
        }

    def standardize_vars(self, vars_list):
        """Standardize variables"""
        scaler = StandardScaler()
        df_scaled = self.df.copy()
        df_scaled[vars_list] = scaler.fit_transform(self.df[vars_list])
        return df_scaled

    def mediation_test(self, X, M, Y, df):
        """
        Mediation analysis Baron & Kenny four-step method

        Args:
            X: Independent variable (BFI trait)
            M: Mediator variable (LIWC feature)
            Y: Dependent variable (Behavior)
            df: DataFrame

        Returns:
            dict: Mediation analysis results
        """
        results = {}

        # Step 1: Total effect (c path): X → Y
        formula_total = f"{Y} ~ {X}"
        model_total = smf.ols(formula_total, df).fit()
        c = model_total.params[X]
        c_pval = model_total.pvalues[X]

        results['step1_total_effect'] = {
            'c': c,
            'pval': c_pval,
            'significant': c_pval < 0.05
        }

        # Step 2: Path a: X → M
        formula_a = f"{M} ~ {X}"
        model_a = smf.ols(formula_a, df).fit()
        a = model_a.params[X]
        a_pval = model_a.pvalues[X]

        results['step2_path_a'] = {
            'a': a,
            'pval': a_pval,
            'significant': a_pval < 0.05
        }

        # Step 3: Path b: M → Y (controlling X)
        formula_b = f"{Y} ~ {X} + {M}"
        model_b = smf.ols(formula_b, df).fit()
        b = model_b.params[M]
        b_pval = model_b.pvalues[M]
        c_prime = model_b.params[X]  # Direct effect
        c_prime_pval = model_b.pvalues[X]

        results['step3_path_b'] = {
            'b': b,
            'pval': b_pval,
            'significant': b_pval < 0.05
        }

        # Step 4: Direct effect (c' path): X → Y (controlling M)
        results['step4_direct_effect'] = {
            'c_prime': c_prime,
            'pval': c_prime_pval,
            'significant': c_prime_pval < 0.05
        }

        # Indirect effect = a × b
        indirect_effect = a * b

        # Sobel test for indirect effect
        se_a = model_a.bse[X]
        se_b = model_b.bse[M]
        sobel_se = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
        sobel_z = indirect_effect / sobel_se if sobel_se > 0 else 0
        sobel_pval = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

        results['indirect_effect'] = {
            'effect': indirect_effect,
            'sobel_z': sobel_z,
            'sobel_pval': sobel_pval,
            'significant': sobel_pval < 0.05
        }

        # Mediation type判断
        if results['step2_path_a']['significant'] and results['step3_path_b']['significant']:
            if not results['step4_direct_effect']['significant']:
                mediation_type = 'Full mediation'
            else:
                if np.sign(c) == np.sign(c_prime):
                    mediation_type = 'Partial mediation'
                else:
                    mediation_type = 'Inconsistent mediation'
        else:
            mediation_type = 'No mediation'

        results['mediation_type'] = mediation_type

        # Proportion mediated
        if c != 0:
            prop_mediated = (c - c_prime) / c
        else:
            prop_mediated = 0

        results['proportion_mediated'] = prop_mediated

        return results

    def bootstrap_indirect_effect(self, X, M, Y, df, n_bootstrap=5000):
        """
        Bootstrap confidence interval calculation indirect effect (Preacher & Hayes)
        """
        def compute_indirect(data_indices):
            """Compute indirect effect for a single bootstrap sample"""
            sample_df = df.iloc[data_indices]

            # Path a: X → M
            try:
                model_a = smf.ols(f"{M} ~ {X}", sample_df).fit()
                a = model_a.params[X]

                # Path b: M → Y (controlling X)
                model_b = smf.ols(f"{Y} ~ {X} + {M}", sample_df).fit()
                b = model_b.params[M]

                return a * b
            except:
                return 0

        # Execute bootstrap
        n = len(df)
        indirect_effects = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            ie = compute_indirect(indices)
            indirect_effects.append(ie)

        indirect_effects = np.array(indirect_effects)

        # Calculate 95% CI (percentile method)
        ci_lower = np.percentile(indirect_effects, 2.5)
        ci_upper = np.percentile(indirect_effects, 97.5)

        # Significance: CI does not include 0
        significant = not (ci_lower <= 0 <= ci_upper)

        return {
            'mean': np.mean(indirect_effects),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': significant
        }

    def run_mediation_for_triplet(self, bfi_trait, liwc_feature, behavior):
        """Run mediation analysis for a single triplet"""
        print(f"\n{'='*60}")
        print(f"Triplet: {bfi_trait} → {liwc_feature} → {behavior}")
        print(f"{'='*60}")

        # Standardize
        vars_to_scale = [bfi_trait, liwc_feature, behavior]
        df_scaled = self.standardize_vars(vars_to_scale)

        # Overall mediation (Entire sample)
        print("\n[Overall Analysis]")
        results_overall = self.mediation_test(bfi_trait, liwc_feature, behavior, df_scaled)

        print(f"  Step 1 - Total effect (c): {results_overall['step1_total_effect']['c']:+.3f}, p={results_overall['step1_total_effect']['pval']:.4f}")
        print(f"  Step 2 - Path a (X→M): {results_overall['step2_path_a']['a']:+.3f}, p={results_overall['step2_path_a']['pval']:.4f}")
        print(f"  Step 3 - Path b (M→Y|X): {results_overall['step3_path_b']['b']:+.3f}, p={results_overall['step3_path_b']['pval']:.4f}")
        print(f"  Step 4 - Direct effect (c'): {results_overall['step4_direct_effect']['c_prime']:+.3f}, p={results_overall['step4_direct_effect']['pval']:.4f}")
        print(f"  Indirect effect (a×b): {results_overall['indirect_effect']['effect']:+.3f}, Sobel p={results_overall['indirect_effect']['sobel_pval']:.4f}")
        print(f"  Mediation type: {results_overall['mediation_type']}")
        print(f"  Proportion mediated: {results_overall['proportion_mediated']:.1%}")

        # Bootstrap CI
        print(f"\n  Bootstrap 95% CI (n=5000):")
        bootstrap_results = self.bootstrap_indirect_effect(bfi_trait, liwc_feature, behavior, df_scaled)
        print(f"    Indirect effect: {bootstrap_results['mean']:+.3f} [{bootstrap_results['ci_lower']:+.3f}, {bootstrap_results['ci_upper']:+.3f}]")
        print(f"    Significant: {bootstrap_results['significant']}")

        # By-model analysis
        print("\n[By-Model Analysis]")
        model_results = []

        for model in sorted(self.df['model'].unique()):
            df_model = df_scaled[df_scaled['model'] == model]

            if len(df_model) >= 10:  # At least 10 observations
                res = self.mediation_test(bfi_trait, liwc_feature, behavior, df_model)

                model_results.append({
                    'model': model,
                    'n': len(df_model),
                    'a': res['step2_path_a']['a'],
                    'b': res['step3_path_b']['b'],
                    'indirect': res['indirect_effect']['effect'],
                    'sobel_p': res['indirect_effect']['sobel_pval'],
                    'mediation_type': res['mediation_type'],
                    'prop_mediated': res['proportion_mediated']
                })

                print(f"  {model}: a={res['step2_path_a']['a']:+.3f}, b={res['step3_path_b']['b']:+.3f}, "
                      f"indirect={res['indirect_effect']['effect']:+.3f}, p={res['indirect_effect']['sobel_pval']:.4f}")

        return {
            'triplet': f"{bfi_trait}→{liwc_feature}→{behavior}",
            'overall': results_overall,
            'bootstrap': bootstrap_results,
            'by_model': pd.DataFrame(model_results)
        }

    def run_all_mediations(self):
        """Run all mediation analysis"""
        print("\n" + "="*60)
        print("RQ2: Mediation Analysis - BFI → LIWC → Behavior")
        print("="*60)

        all_results = {}

        for behavior, triplets in self.mediations.items():
            print(f"\n{'#'*60}")
            print(f"# Behavior: {behavior}")
            print(f"{'#'*60}")

            behavior_results = []

            for bfi_trait, liwc_feature, expectation in triplets:
                result = self.run_mediation_for_triplet(bfi_trait, liwc_feature, behavior)
                result['expectation'] = expectation
                behavior_results.append(result)

            all_results[behavior] = behavior_results

        return all_results

    def visualize_mediation_paths(self, all_results):
        """Visualize mediation paths"""
        print("\nGenerating mediation path plots...")

        for behavior, results in all_results.items():
            n_triplets = len(results)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            for idx, result in enumerate(results[:5]):  # At most 5 triplets
                ax = axes[idx]

                triplet = result['triplet']
                overall = result['overall']

                # Extract path coefficients
                a = overall['step2_path_a']['a']
                b = overall['step3_path_b']['b']
                c = overall['step1_total_effect']['c']
                c_prime = overall['step4_direct_effect']['c_prime']
                indirect = overall['indirect_effect']['effect']

                # Draw mediation model
                # Node positions
                x_positions = {'X': 0, 'M': 1, 'Y': 2}
                y_positions = {'X': 0, 'M': 0.5, 'Y': 0}

                # Draw nodes
                for node, x in x_positions.items():
                    y = y_positions[node]
                    circle = plt.Circle((x, y), 0.15, color='lightblue', ec='black', linewidth=2, zorder=10)
                    ax.add_patch(circle)
                    label = triplet.split('→')[0] if node == 'X' else (triplet.split('→')[1] if node == 'M' else triplet.split('→')[2])
                    ax.text(x, y, label.replace('_', '\n'), ha='center', va='center', fontsize=9, fontweight='bold', zorder=11)

                # Draw path arrows
                # Path a: X → M
                ax.annotate('', xy=(x_positions['M']-0.15, y_positions['M']), xytext=(x_positions['X']+0.15, y_positions['X']),
                           arrowprops=dict(arrowstyle='->', lw=2, color='green' if abs(a) > 0.1 else 'gray'))
                ax.text(0.5, 0.3, f"a={a:+.3f}{'*' if overall['step2_path_a']['significant'] else ''}", ha='center', fontsize=10, color='green' if abs(a) > 0.1 else 'gray')

                # Path b: M → Y
                ax.annotate('', xy=(x_positions['Y']-0.15, y_positions['Y']), xytext=(x_positions['M']+0.15, y_positions['M']),
                           arrowprops=dict(arrowstyle='->', lw=2, color='green' if abs(b) > 0.1 else 'gray'))
                ax.text(1.5, 0.3, f"b={b:+.3f}{'*' if overall['step3_path_b']['significant'] else ''}", ha='center', fontsize=10, color='green' if abs(b) > 0.1 else 'gray')

                # Direct effect c': X → Y (dashed)
                ax.annotate('', xy=(x_positions['Y']-0.15, y_positions['Y']-0.05), xytext=(x_positions['X']+0.15, y_positions['X']-0.05),
                           arrowprops=dict(arrowstyle='->', lw=1.5, linestyle='--', color='red' if abs(c_prime) > 0.1 else 'lightgray'))
                ax.text(1, -0.25, f"c'={c_prime:+.3f}{'*' if overall['step4_direct_effect']['significant'] else ''}", ha='center', fontsize=9, color='red' if abs(c_prime) > 0.1 else 'gray')

                # Add total effect and indirect effect
                ax.text(1, 0.8, f"Total (c): {c:+.3f}{'*' if overall['step1_total_effect']['significant'] else ''}", ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax.text(1, 0.65, f"Indirect (a×b): {indirect:+.3f}{'*' if result['bootstrap']['significant'] else ''}", ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen' if result['bootstrap']['significant'] else 'lightgray', alpha=0.5))

                # Mediation type
                mediation_type = overall['mediation_type']
                ax.text(1, -0.5, f"Type: {mediation_type}", ha='center', fontsize=10, fontweight='bold',
                       color='green' if 'mediation' in mediation_type.lower() and 'No' not in mediation_type else 'red')

                ax.set_xlim(-0.3, 2.3)
                ax.set_ylim(-0.7, 1.0)
                ax.axis('off')
                ax.set_title(triplet, fontsize=11, fontweight='bold')

            plt.suptitle(f"Mediation Paths for {behavior.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.fig_dir / f'mediation_paths_{behavior}.png', dpi=300, bbox_inches='tight')
            print(f"  ✓ mediation_paths_{behavior}.png")
            plt.close()

    def generate_summary_table(self, all_results):
        """Generate mediation analysis summary table"""
        print("\nGenerating summary table...")

        summary_data = []

        for behavior, results in all_results.items():
            for result in results:
                triplet = result['triplet']
                overall = result['overall']
                bootstrap = result['bootstrap']

                summary_data.append({
                    'Behavior': behavior.replace('_', ' ').title(),
                    'Triplet': triplet,
                    'Path_a': overall['step2_path_a']['a'],
                    'Path_a_p': overall['step2_path_a']['pval'],
                    'Path_b': overall['step3_path_b']['b'],
                    'Path_b_p': overall['step3_path_b']['pval'],
                    'Indirect_effect': overall['indirect_effect']['effect'],
                    'Bootstrap_CI_lower': bootstrap['ci_lower'],
                    'Bootstrap_CI_upper': bootstrap['ci_upper'],
                    'Significant': bootstrap['significant'],
                    'Mediation_type': overall['mediation_type'],
                    'Proportion_mediated': overall['proportion_mediated']
                })

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(self.output_dir / 'rq2_mediation_summary.csv', index=False)
        print(f"  ✓ rq2_mediation_summary.csv")

        return df_summary


def main():
    """Main function"""
    analyzer = MediationAnalysis()

    # Run all mediation analysis
    all_results = analyzer.run_all_mediations()

    # Generate visualization
    analyzer.visualize_mediation_paths(all_results)

    # Generate summary table
    df_summary = analyzer.generate_summary_table(all_results)

    print("\n" + "="*60)
    print("✅ RQ2 mediation analysis completed!")
    print("="*60)

    # Print key findings
    print("\nKey findings summary:")
    print(f"  Total {len(df_summary)} mediation triplets")
    print(f"  Significant indirect effects: {df_summary['Significant'].sum()} 个")
    print(f"  Full mediation: {(df_summary['Mediation_type'] == 'Full mediation').sum()} 个")
    print(f"  Partial mediation: {(df_summary['Mediation_type'] == 'Partial mediation').sum()} 个")
    print(f"  No mediation: {(df_summary['Mediation_type'] == 'No mediation').sum()} 个")

    print("\nOutput files:")
    print("  - data/analysis/rq2_mediation_summary.csv")
    print("  - data/analysis/figures/mediation_paths_risk_mean_cards.png")
    print("  - data/analysis/figures/mediation_paths_sycophancy_rate.png")


if __name__ == '__main__':
    main()
