"""
Regression analysis main script
Implement RQ1-RQ3 statistical analysis

RQ1: Prediction validity - LIWC vs BFI predict behavior
RQ2: Cross-prompt stability (not implemented, because temperature=0 single run)
RQ3: Differential validity - which LIWC categories predict which behaviors

Output:
- data/analysis/rq1_model_comparison.csv
- data/analysis/rq3_liwc_behavior_heatmap.csv
- data/analysis/figures/*.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class RegressionAnalysis:
    def __init__(self, data_path='data/analysis/merged_data.csv'):
        """Initialize analysis"""
        self.df = pd.read_csv(data_path)
        self.output_dir = Path('data/analysis')
        self.output_dir.mkdir(exist_ok=True)
        self.fig_dir = self.output_dir / 'figures'
        self.fig_dir.mkdir(exist_ok=True)

        # Drop missing values
        self.df = self.df.dropna()
        print(f"Loaded data: {len(self.df)} rows (after removing missing values)")

        # Define variable groups
        self.bfi_traits = ['openness', 'conscientiousness', 'extraversion',
                           'agreeableness', 'neuroticism']
        self.behaviors = ['risk_mean_cards', 'sycophancy_rate']

        # Select key LIWC features (based on expected associations in the proposal)
        self.liwc_features = [
            # Affective processes
            'tone_pos', 'tone_neg', 'emo_anx', 'emo_anger', 'emo_sad',
            # Social processes
            'Social', 'prosocial', 'conflict', 'socrefs',
            # Cognitive processes
            'insight', 'cause', 'certitude', 'tentat',
            # Personal concerns
            'achieve', 'power', 'risk', 'reward',
            # General
            'Analytic', 'Clout', 'Authentic', 'Tone',
            'i', 'we', 'you'  # pronouns
        ]

        # Ensure all features exist
        self.liwc_features = [f for f in self.liwc_features if f in self.df.columns]

        print(f"BFI features: {len(self.bfi_traits)}")
        print(f"LIWC features: {len(self.liwc_features)}")
        print(f"Behavior variables: {len(self.behaviors)}")

    def standardize_features(self, features):
        """Standardize features"""
        scaler = StandardScaler()
        return pd.DataFrame(
            scaler.fit_transform(self.df[features]),
            columns=features,
            index=self.df.index
        )

    def fit_models_for_behavior(self, behavior):
        """
        Fit 4 models for a single behavior variable
        M0: Null (intercept only)
        M1: BFI only
        M2: LIWC only
        M3: BFI + LIWC
        """
        results = {}

        # Prepare data
        y = self.df[behavior].values
        X_bfi = self.standardize_features(self.bfi_traits)
        X_liwc = self.standardize_features(self.liwc_features)

        # M0: Null model (intercept only)
        X0 = sm.add_constant(np.ones(len(y)))
        m0 = sm.OLS(y, X0).fit()
        results['M0_null'] = {
            'r2': m0.rsquared,
            'r2_adj': m0.rsquared_adj,
            'aic': m0.aic,
            'bic': m0.bic,
            'n_params': len(m0.params)
        }

        # M1: BFI only
        X1 = sm.add_constant(X_bfi)
        m1 = sm.OLS(y, X1).fit()
        results['M1_bfi'] = {
            'r2': m1.rsquared,
            'r2_adj': m1.rsquared_adj,
            'aic': m1.aic,
            'bic': m1.bic,
            'n_params': len(m1.params),
            'sig_predictors': sum(m1.pvalues[1:] < 0.05)  # exclude intercept
        }

        # M2: LIWC only
        X2 = sm.add_constant(X_liwc)
        m2 = sm.OLS(y, X2).fit()
        results['M2_liwc'] = {
            'r2': m2.rsquared,
            'r2_adj': m2.rsquared_adj,
            'aic': m2.aic,
            'bic': m2.bic,
            'n_params': len(m2.params),
            'sig_predictors': sum(m2.pvalues[1:] < 0.05)
        }

        # M3: BFI + LIWC
        X3 = sm.add_constant(pd.concat([X_bfi, X_liwc], axis=1))
        m3 = sm.OLS(y, X3).fit()
        results['M3_combined'] = {
            'r2': m3.rsquared,
            'r2_adj': m3.rsquared_adj,
            'aic': m3.aic,
            'bic': m3.bic,
            'n_params': len(m3.params),
            'sig_predictors': sum(m3.pvalues[1:] < 0.05)
        }

        # Model comparison tests
        # M1 vs M0
        f_stat_1v0 = ((m0.ssr - m1.ssr) / (m1.df_model - m0.df_model)) / (m1.ssr / m1.df_resid)
        p_value_1v0 = 1 - stats.f.cdf(f_stat_1v0, m1.df_model - m0.df_model, m1.df_resid)

        # M2 vs M0
        f_stat_2v0 = ((m0.ssr - m2.ssr) / (m2.df_model - m0.df_model)) / (m2.ssr / m2.df_resid)
        p_value_2v0 = 1 - stats.f.cdf(f_stat_2v0, m2.df_model - m0.df_model, m2.df_resid)

        # M2 vs M1
        if m2.rsquared > m1.rsquared:
            f_stat_2v1 = ((m1.ssr - m2.ssr) / abs(m2.df_model - m1.df_model)) / (m2.ssr / m2.df_resid)
            p_value_2v1 = 1 - stats.f.cdf(f_stat_2v1, abs(m2.df_model - m1.df_model), m2.df_resid)
        else:
            f_stat_2v1 = np.nan
            p_value_2v1 = np.nan

        results['comparisons'] = {
            'M1_vs_M0_f': f_stat_1v0,
            'M1_vs_M0_p': p_value_1v0,
            'M2_vs_M0_f': f_stat_2v0,
            'M2_vs_M0_p': p_value_2v0,
            'M2_vs_M1_f': f_stat_2v1,
            'M2_vs_M1_p': p_value_2v1,
            'delta_r2_M2_M1': m2.rsquared - m1.rsquared
        }

        # Save model objects for further analysis
        results['models'] = {'M0': m0, 'M1': m1, 'M2': m2, 'M3': m3}

        return results

    def run_rq1_analysis(self):
        """
        RQ1: Prediction validity analysis
        Compare LIWC vs BFI in predicting behavior
        """
        print("\n" + "="*60)
        print("RQ1: Predictive Validity Analysis")
        print("="*60)

        all_results = []

        for behavior in self.behaviors:
            print(f"\n--- Analyze behavior: {behavior} ---")
            results = self.fit_models_for_behavior(behavior)

            # Print results
            print("\nModel Performance:")
            for model_name in ['M0_null', 'M1_bfi', 'M2_liwc', 'M3_combined']:
                r = results[model_name]
                print(f"{model_name:15} R²={r['r2']:.4f}, R²_adj={r['r2_adj']:.4f}, "
                      f"AIC={r['aic']:.1f}, BIC={r['bic']:.1f}")

            print("\nModel Comparisons:")
            comp = results['comparisons']
            print(f"M1 vs M0: F={comp['M1_vs_M0_f']:.2f}, p={comp['M1_vs_M0_p']:.4f}")
            print(f"M2 vs M0: F={comp['M2_vs_M0_f']:.2f}, p={comp['M2_vs_M0_p']:.4f}")
            print(f"M2 vs M1: ΔR²={comp['delta_r2_M2_M1']:.4f}, "
                  f"F={comp.get('M2_vs_M1_f', np.nan):.2f}, "
                  f"p={comp.get('M2_vs_M1_p', np.nan):.4f}")

            # Save results
            for model_name in ['M0_null', 'M1_bfi', 'M2_liwc', 'M3_combined']:
                row = {
                    'behavior': behavior,
                    'model': model_name,
                    **results[model_name]
                }
                all_results.append(row)

        # Save summary table
        results_df = pd.DataFrame(all_results)
        output_file = self.output_dir / 'rq1_model_comparison.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        # Visualize
        self.plot_rq1_results(results_df)

        return results_df

    def plot_rq1_results(self, results_df):
        """Visualize RQ1 results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('RQ1: Model Performance Comparison', fontsize=16, y=1.02)

        metrics = ['r2', 'r2_adj', 'aic', 'bic']
        titles = ['R²', 'Adjusted R²', 'AIC (lower is better)', 'BIC (lower is better)']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            # Prepare data
            plot_df = results_df.pivot(index='behavior', columns='model', values=metric)
            plot_df = plot_df[['M0_null', 'M1_bfi', 'M2_liwc', 'M3_combined']]

            # Plot
            plot_df.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel(metric.upper())
            ax.legend(title='Model', loc='best')
            ax.grid(axis='y', alpha=0.3)

            # Rotate x-axis labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        fig_file = self.fig_dir / 'rq1_model_comparison.png'
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {fig_file}")
        plt.close()

    def run_rq3_analysis(self):
        """
        RQ3: Differential Validity Analysis
        Generate correlation matrix and heatmap for LIWC features × behaviors
        """
        print("\n" + "="*60)
        print("RQ3: Differential Validity Analysis")
        print("="*60)

        # Calculate correlation coefficients and significance
        correlations = []

        for liwc_feat in self.liwc_features:
            for behavior in self.behaviors:
                # Drop missing values
                valid_idx = self.df[[liwc_feat, behavior]].dropna().index
                if len(valid_idx) < 10:
                    continue

                x = self.df.loc[valid_idx, liwc_feat]
                y = self.df.loc[valid_idx, behavior]

                # Calculate Pearson correlation
                r, p = stats.pearsonr(x, y)

                correlations.append({
                    'liwc_feature': liwc_feat,
                    'behavior': behavior,
                    'correlation': r,
                    'p_value': p,
                    'significant': p < 0.05,
                    'n': len(valid_idx)
                })

        corr_df = pd.DataFrame(correlations)

        # Save results
        output_file = self.output_dir / 'rq3_liwc_behavior_correlations.csv'
        corr_df.to_csv(output_file, index=False)
        print(f"\nCorrelation analysis results saved: {output_file}")

        # Print number of significant correlations
        print(f"\n总计 {len(corr_df)} 个LIWC-Behavior配对")
        print(f"显著相关 (p<0.05): {corr_df['significant'].sum()} 个 "
              f"({100*corr_df['significant'].sum()/len(corr_df):.1f}%)")

        # 按行为分组统计
        print("\n各行为的显著相关LIWC特征数量:")
        for behavior in self.behaviors:
            n_sig = corr_df[(corr_df['behavior'] == behavior) &
                           (corr_df['significant'])].shape[0]
            total = corr_df[corr_df['behavior'] == behavior].shape[0]
            print(f"  {behavior}: {n_sig}/{total} ({100*n_sig/total:.1f}%)")

        # 可视化热图
        self.plot_rq3_heatmap(corr_df)

        return corr_df

    def plot_rq3_heatmap(self, corr_df):
        """Visualize LIWC × Behavior heatmap"""
        # Create correlation matrix
        corr_matrix = corr_df.pivot(index='liwc_feature',
                                     columns='behavior',
                                     values='correlation')

        # Create significance matrix
        sig_matrix = corr_df.pivot(index='liwc_feature',
                                   columns='behavior',
                                   values='significant')

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 14))

        # Use diverging colormap
        sns.heatmap(corr_matrix,
                   annot=True,
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0,
                   vmin=-0.5, vmax=0.5,
                   square=False,
                   cbar_kws={'label': 'Pearson r'},
                   ax=ax)

        # Mark significant correlations
        for i, liwc_feat in enumerate(corr_matrix.index):
            for j, behavior in enumerate(corr_matrix.columns):
                if sig_matrix.loc[liwc_feat, behavior]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                              fill=False, edgecolor='black',
                                              lw=2))

        ax.set_title('RQ3: LIWC Features × Behaviors Correlation\n(Black boxes indicate p < 0.05)',
                    fontsize=14, pad=20)
        ax.set_xlabel('Behavior', fontsize=12)
        ax.set_ylabel('LIWC Feature', fontsize=12)

        plt.tight_layout()
        fig_file = self.fig_dir / 'rq3_liwc_behavior_heatmap.png'
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {fig_file}")
        plt.close()

    def generate_summary_report(self, rq1_df, rq3_df):
        """Generate analysis summary report"""
        report_file = self.output_dir / 'analysis_summary.txt'

        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("LLM Personality Triangulation - Analysis Summary\n")
            f.write("="*70 + "\n\n")

            f.write("RQ1: Predictive Validity (LIWC vs BFI)\n")
            f.write("-"*70 + "\n")

            for behavior in self.behaviors:
                f.write(f"\n{behavior}:\n")
                behavior_results = rq1_df[rq1_df['behavior'] == behavior]

                m1 = behavior_results[behavior_results['model'] == 'M1_bfi'].iloc[0]
                m2 = behavior_results[behavior_results['model'] == 'M2_liwc'].iloc[0]

                f.write(f"  BFI (M1):  R² = {m1['r2']:.4f}, AIC = {m1['aic']:.1f}\n")
                f.write(f"  LIWC (M2): R² = {m2['r2']:.4f}, AIC = {m2['aic']:.1f}\n")
                f.write(f"  ΔR² = {m2['r2'] - m1['r2']:.4f}\n")

                if m2['r2'] > m1['r2']:
                    f.write(f"  → LIWC predicts better than BFI\n")
                else:
                    f.write(f"  → BFI predicts better than LIWC\n")

            f.write("\n" + "="*70 + "\n")
            f.write("RQ3: Differential Validity (LIWC-Behavior Associations)\n")
            f.write("-"*70 + "\n\n")

            total_pairs = len(rq3_df)
            sig_pairs = rq3_df['significant'].sum()
            f.write(f"Total LIWC-Behavior pairs tested: {total_pairs}\n")
            f.write(f"Significant associations (p<0.05): {sig_pairs} ({100*sig_pairs/total_pairs:.1f}%)\n\n")

            f.write("Top 10 Strongest Positive Associations:\n")
            top_pos = rq3_df.nlargest(10, 'correlation')
            for _, row in top_pos.iterrows():
                sig = "*" if row['significant'] else ""
                f.write(f"  {row['liwc_feature']:20} → {row['behavior']:20} "
                       f"r={row['correlation']:6.3f} {sig}\n")

            f.write("\nTop 10 Strongest Negative Associations:\n")
            top_neg = rq3_df.nsmallest(10, 'correlation')
            for _, row in top_neg.iterrows():
                sig = "*" if row['significant'] else ""
                f.write(f"  {row['liwc_feature']:20} → {row['behavior']:20} "
                       f"r={row['correlation']:6.3f} {sig}\n")

        print(f"\nSummary report saved to: {report_file}")

    def run_all(self):
        """Run all analysis"""
        print("\nStarting complete analysis...")

        # RQ1
        rq1_results = self.run_rq1_analysis()

        # RQ3
        rq3_results = self.run_rq3_analysis()

        # Generate summary report
        self.generate_summary_report(rq1_results, rq3_results)

        print("\n" + "="*70)
        print("Analysis completed!")
        print("="*70)
        print(f"\nAll results saved to: {self.output_dir}/")
        print(f"All figures saved to: {self.fig_dir}/")


def main():
    # Run analysis
    analysis = RegressionAnalysis()
    analysis.run_all()


if __name__ == '__main__':
    main()
