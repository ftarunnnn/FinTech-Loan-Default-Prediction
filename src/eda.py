import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def run_eda(input_path='data/processed/clean_loan_data.csv', output_dir='artifacts/eda_plots'):
    """
    Phase 4: EDA & Financial Behavior Analysis
    Generates exploratory visual plots and statistical summaries.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    
    print(f"[EDA] Running EDA on clean dataset: {input_path}")
    
    # 1. Target Class Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    target_counts = df['loan_status'].value_counts()
    sns.barplot(x=['Fully Paid (0)', 'Default (1)'], y=target_counts.values, palette=['#2ecc71', '#e74c3c'], ax=ax)
    ax.set_title('Loan Default Class Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Applicants')
    for i, count in enumerate(target_counts.values):
        pct = (count / len(df)) * 100
        ax.text(i, count / 2, f"{count:,}\n({pct:.1f}%)", ha='center', color='white', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved plot: {plot1_path}")
    
    # 2. Univariate Analysis (Income, Credit Score, Loan Amount, Interest Rate)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.histplot(df['annual_income'], kde=True, color='#3498db', ax=axes[0, 0])
    axes[0, 0].set_title('Annual Income Distribution ($)', fontweight='bold')
    
    sns.histplot(df['credit_score'], kde=True, color='#9b59b6', ax=axes[0, 1])
    axes[0, 1].set_title('Credit Score Distribution (FICO)', fontweight='bold')
    
    sns.histplot(df['loan_amount'], kde=True, color='#1abc9c', ax=axes[1, 0])
    axes[1, 0].set_title('Loan Amount Distribution ($)', fontweight='bold')
    
    sns.histplot(df['interest_rate'], kde=True, color='#e67e22', ax=axes[1, 1])
    axes[1, 1].set_title('Interest Rate Distribution (%)', fontweight='bold')
    
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, 'univariate_distributions.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved plot: {plot2_path}")
    
    # 3. Bivariate Behavior Analysis vs Default
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sns.boxplot(x='loan_status', y='dti', data=df, palette=['#2ecc71', '#e74c3c'], ax=axes[0, 0])
    axes[0, 0].set_title('Debt-to-Income (DTI) vs Default Status', fontweight='bold')
    axes[0, 0].set_xticklabels(['Fully Paid', 'Default'])
    
    sns.boxplot(x='loan_status', y='credit_score', data=df, palette=['#2ecc71', '#e74c3c'], ax=axes[0, 1])
    axes[0, 1].set_title('Credit Score vs Default Status', fontweight='bold')
    axes[0, 1].set_xticklabels(['Fully Paid', 'Default'])
    
    sns.boxplot(x='loan_status', y='annual_income', data=df, palette=['#2ecc71', '#e74c3c'], ax=axes[1, 0])
    axes[1, 0].set_title('Annual Income vs Default Status', fontweight='bold')
    axes[1, 0].set_xticklabels(['Fully Paid', 'Default'])
    
    sns.boxplot(x='loan_status', y='interest_rate', data=df, palette=['#2ecc71', '#e74c3c'], ax=axes[1, 1])
    axes[1, 1].set_title('Interest Rate vs Default Status', fontweight='bold')
    axes[1, 1].set_xticklabels(['Fully Paid', 'Default'])
    
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, 'bivariate_insights.png')
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved plot: {plot3_path}")
    
    # 4. Correlation Heatmap
    num_cols = ['annual_income', 'credit_score', 'dti', 'revol_bal', 'total_credit_lines', 
                'open_credit_lines', 'public_derogatory', 'inquiries_last_6m', 'loan_amount', 
                'interest_rate', 'loan_status']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Financial Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot4_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved plot: {plot4_path}")
    
    print("[SUCCESS] Phase 4 EDA Visualizations Complete!")

if __name__ == '__main__':
    run_eda()
