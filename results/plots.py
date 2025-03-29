import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def get_model_colors(model_order):
    """Generate a consistent color mapping for the given models."""
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return {model: colors[i % len(colors)] for i, model in enumerate(model_order)}

def load_data(data_dir):
    dfs = []
    for file in os.listdir(data_dir):
        if file.endswith('__dev.parquet'):
            print("loading", file)
            model = file.split('__')[0]
            df = pd.read_parquet(os.path.join(data_dir, file))
            df['model'] = model
            dfs.append(df)
    df = pd.concat(dfs)
    # Get unique models and sort them for consistent ordering
    model_order = sorted(df['model'].unique().tolist())
    model_colors = get_model_colors(model_order)
    return df, model_order, model_colors

def plot_model_performance(df, model_order, model_colors):
    decision_df = df[df['type'] == 'decision'].copy()
    decision_df['correct'] = (decision_df['distill_answer'] == decision_df['complete_answer'])
    
    model_stats = []
    for model in model_order:
        model_data = decision_df[decision_df['model'] == model]['correct']
        mean = model_data.mean()
        ci = stats.t.interval(0.9, len(model_data) - 1, loc=mean, scale=stats.sem(model_data))
        model_stats.append({
            'model': model,
            'mean': mean,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        })
    
    stats_df = pd.DataFrame(model_stats)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        stats_df['model'], stats_df['mean'], 
        color=[model_colors[model] for model in stats_df['model']],
        alpha=0.7
    )
    
    plt.errorbar(
        stats_df['model'],
        stats_df['mean'],
        yerr=[
            stats_df['mean'] - stats_df['ci_lower'], 
            stats_df['ci_upper'] - stats_df['mean']
        ],
        fmt='none',
        color='black',
        capsize=5
    )
    
    plt.title('HoVer accuracy across models')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('figs/hover_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_cumulative_accuracy(df, model_order, model_colors):
    response_df = df[df['type'] == 'response'].copy()
    response_df['correct'] = (response_df['distill_answer'] == response_df['complete_answer'])
    response_df['iteration'] = response_df.groupby(['model', 'id']).cumcount() + 1
    
    model_stats = []
    for model in model_order:
        model_data = response_df[response_df['model'] == model]
        for iteration in model_data['iteration'].unique():
            iter_data = model_data[model_data['iteration'] == iteration]['correct']
            mean = iter_data.mean()
            ci = stats.t.interval(0.9, len(iter_data) - 1, loc=mean, scale=stats.sem(iter_data))
            model_stats.append({
                'model': model,
                'iteration': iteration,
                'mean': mean,
                'ci_lower': ci[0],
                'ci_upper': ci[1]
            })
    stats_df = pd.DataFrame(model_stats)
    
    plt.figure(figsize=(10, 6))
    for model in model_order:
        model_data = stats_df[stats_df['model'] == model]
        plt.plot(model_data['iteration'], model_data['mean'], 
                label=model, marker='o', color=model_colors[model])
        plt.fill_between(
            model_data['iteration'],
            model_data['ci_lower'],
            model_data['ci_upper'],
            alpha=0.2,
            color=model_colors[model]
        )
    
    plt.title('Accuracy gain as more tool calls allowed')
    plt.xlabel('Number of hops')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig('figs/cumulative_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_initial_final_agreement(df, model_order, model_colors):
    response_df = df[df['type'] == 'response'].copy()
    decision_df = df[df['type'] == 'decision'].copy()
    
    initial_responses = response_df.groupby(['model', 'id']).first().reset_index()
    
    comparison_df = pd.merge(
        initial_responses[['model', 'id', 'distill_answer']],
        decision_df[['model', 'id', 'distill_answer']],
        on=['model', 'id'],
        suffixes=('_initial', '_final')
    )
    
    comparison_df['agrees'] = (comparison_df['distill_answer_initial'] == comparison_df['distill_answer_final'])
    
    model_stats = []
    for model in model_order:
        model_data = comparison_df[comparison_df['model'] == model]['agrees']
        mean = model_data.mean()
        ci = stats.t.interval(0.9, len(model_data) - 1, loc=mean, scale=stats.sem(model_data))
        model_stats.append({
            'model': model,
            'mean': mean,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        })
    
    stats_df = pd.DataFrame(model_stats)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        stats_df['model'], stats_df['mean'],
        color=[model_colors[model] for model in stats_df['model']],
        alpha=0.7
    )
    
    plt.errorbar(
        stats_df['model'],
        stats_df['mean'],
        yerr=[
            stats_df['mean'] - stats_df['ci_lower'],
            stats_df['ci_upper'] - stats_df['mean']
        ],
        fmt='none',
        color='black',
        capsize=5
    )
    
    plt.title('Agreement between initial response and final decision')
    plt.xlabel('Model')
    plt.ylabel('Agreement rate')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('figs/initial_final_agreement.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_decision_distribution(df, model_order, model_colors):
    decision_df = df[df['type'] == 'decision'].copy()
    
    model_stats = []
    for model in model_order:
        model_data = decision_df[decision_df['model'] == model]
        total = len(model_data)
        
        valid_answers = ((model_data['distill_answer'] == 'SUPPORTED') | 
                        (model_data['distill_answer'] == 'NOT_SUPPORTED')).sum()
        
        valid_ci = stats.binom.interval(0.9, total, valid_answers/total)
        
        model_stats.append({
            'model': model,
            'valid': valid_answers/total,
            'ci_lower': valid_ci[0]/total,
            'ci_upper': valid_ci[1]/total
        })
    
    stats_df = pd.DataFrame(model_stats)
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(
        stats_df['model'], stats_df['valid'],
        color=[model_colors[model] for model in stats_df['model']],
        alpha=0.7
    )
    
    plt.errorbar(
        stats_df['model'],
        stats_df['valid'],
        yerr=[
            stats_df['valid'] - stats_df['ci_lower'],
            stats_df['ci_upper'] - stats_df['valid']
        ],
        fmt='none',
        color='black',
        capsize=5
    )
    
    plt.title('Fraction of correctly formatted answers (SUPPORTED or NOT_SUPPORTED)')
    plt.xlabel('Model')
    plt.ylabel('Fraction')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('figs/decision_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    df, model_order, model_colors = load_data("../data")
    plot_model_performance(df, model_order, model_colors)
    plot_cumulative_accuracy(df, model_order, model_colors)
    plot_initial_final_agreement(df, model_order, model_colors)
    plot_decision_distribution(df, model_order, model_colors)
