"""
Task 3: Analyse data using NumPy and Pandas (20 marks)
Performs statistical analysis on cleaned HackerNews data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def load_cleaned_data(filename="hackernews_cleaned.csv"):
    """
    Load cleaned CSV data
    
    Args:
        filename (str): Input CSV filename
    
    Returns:
        pd.DataFrame: Loaded DataFrame or None if error
    """
    try:
        df = pd.read_csv(filename)
        print(f"✓ Loaded {len(df)} records from '{filename}'")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def statistical_analysis(df):
    """
    Perform statistical analysis using NumPy and Pandas
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame
    
    Returns:
        dict: Dictionary containing analysis results
    """
    results = {}
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    # 1. Score Analysis
    print("\n1. SCORE ANALYSIS")
    print("-" * 40)
    scores = df['score'].values
    results['score_mean'] = np.mean(scores)
    results['score_median'] = np.median(scores)
    results['score_std'] = np.std(scores)
    results['score_min'] = np.min(scores)
    results['score_max'] = np.max(scores)
    results['score_q25'] = np.percentile(scores, 25)
    results['score_q75'] = np.percentile(scores, 75)
    
    print(f"  Mean Score: {results['score_mean']:.2f}")
    print(f"  Median Score: {results['score_median']:.2f}")
    print(f"  Std Dev: {results['score_std']:.2f}")
    print(f"  Min: {results['score_min']:.0f}")
    print(f"  Max: {results['score_max']:.0f}")
    print(f"  Q1 (25%): {results['score_q25']:.2f}")
    print(f"  Q3 (75%): {results['score_q75']:.2f}")
    print(f"  IQR: {results['score_q75'] - results['score_q25']:.2f}")
    
    # 2. Comments (Descendants) Analysis
    print("\n2. COMMENTS (DESCENDANTS) ANALYSIS")
    print("-" * 40)
    descendants = df['descendants'].values
    results['descendants_mean'] = np.mean(descendants)
    results['descendants_median'] = np.median(descendants)
    results['descendants_std'] = np.std(descendants)
    results['descendants_max'] = np.max(descendants)
    
    print(f"  Mean Comments: {results['descendants_mean']:.2f}")
    print(f"  Median Comments: {results['descendants_median']:.2f}")
    print(f"  Std Dev: {results['descendants_std']:.2f}")
    print(f"  Max Comments: {results['descendants_max']:.0f}")
    
    # 3. Correlation Analysis
    print("\n3. CORRELATION ANALYSIS")
    print("-" * 40)
    correlation = np.corrcoef(df['score'], df['descendants'])[0, 1]
    results['correlation_score_descendants'] = correlation
    print(f"  Correlation (Score vs Comments): {correlation:.4f}")
    
    # 4. Author Analysis
    print("\n4. AUTHOR ANALYSIS")
    print("-" * 40)
    results['total_authors'] = df['by'].nunique()
    results['top_author'] = df['by'].value_counts().index[0]
    results['top_author_count'] = df['by'].value_counts().values[0]
    
    print(f"  Total Unique Authors: {results['total_authors']}")
    print(f"  Top Author: {results['top_author']}")
    print(f"  Top Author Posts: {results['top_author_count']}")
    
    author_avg_score = df.groupby('by')['score'].mean().sort_values(ascending=False)
    print(f"  Top 5 Authors by Average Score:")
    for author, score in author_avg_score.head(5).items():
        print(f"    - {author}: {score:.2f}")
    
    # 5. Domain Analysis
    print("\n5. DOMAIN ANALYSIS")
    print("-" * 40)
    results['total_domains'] = df['domain'].nunique()
    top_domains = df['domain'].value_counts().head(5)
    
    print(f"  Total Unique Domains: {results['total_domains']}")
    print(f"  Top 5 Domains:")
    for domain, count in top_domains.items():
        print(f"    - {domain}: {count} posts")
    
    # 6. Content Performance
    print("\n6. CONTENT PERFORMANCE")
    print("-" * 40)
    high_score_threshold = df['score'].quantile(0.75)
    high_score_posts = df[df['score'] >= high_score_threshold]
    
    results['high_performance_count'] = len(high_score_posts)
    results['high_performance_avg_comments'] = high_score_posts['descendants'].mean()
    results['low_performance_avg_comments'] = df[df['score'] < high_score_threshold]['descendants'].mean()
    
    print(f"  High Performance Posts (top 25%): {results['high_performance_count']}")
    print(f"  Avg Comments on High Score Posts: {results['high_performance_avg_comments']:.2f}")
    print(f"  Avg Comments on Low Score Posts: {results['low_performance_avg_comments']:.2f}")
    
    # 7. Time Analysis
    print("\n7. TIME/HOUR ANALYSIS")
    print("-" * 40)
    hour_stats = df.groupby('hour')['score'].agg(['mean', 'median', 'count'])
    best_hour = hour_stats['mean'].idxmax()
    worst_hour = hour_stats['mean'].idxmin()
    
    results['best_posting_hour'] = best_hour
    results['best_posting_hour_avg_score'] = hour_stats.loc[best_hour, 'mean']
    results['worst_posting_hour'] = worst_hour
    
    print(f"  Best Posting Hour: {best_hour}:00")
    print(f"  Best Hour Avg Score: {results['best_posting_hour_avg_score']:.2f}")
    print(f"  Worst Posting Hour: {worst_hour}:00")
    print(f"  Worst Hour Avg Score: {hour_stats.loc[worst_hour, 'mean']:.2f}")
    
    return results

def calculate_percentiles(df):
    """
    Calculate various percentiles
    
    Args:
        df (pd.DataFrame): DataFrame
    """
    print("\n" + "="*60)
    print("PERCENTILE ANALYSIS")
    print("="*60)
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    
    print("\nScore Percentiles:")
    for p in percentiles:
        value = np.percentile(df['score'], p)
        print(f"  {p:2d}th percentile: {value:8.2f}")
    
    print("\nComments Percentiles:")
    for p in percentiles:
        value = np.percentile(df['descendants'], p)
        print(f"  {p:2d}th percentile: {value:8.2f}")

def identify_outliers(df):
    """
    Identify outliers in data
    
    Args:
        df (pd.DataFrame): DataFrame
    """
    print("\n" + "="*60)
    print("OUTLIER DETECTION (IQR Method)")
    print("="*60)
    
    # Score outliers
    Q1_score = df['score'].quantile(0.25)
    Q3_score = df['score'].quantile(0.75)
    IQR_score = Q3_score - Q1_score
    score_outliers = df[(df['score'] < Q1_score - 1.5 * IQR_score) | 
                        (df['score'] > Q3_score + 1.5 * IQR_score)]
    
    print(f"\nScore Outliers: {len(score_outliers)} ({len(score_outliers)/len(df)*100:.2f}%)")
    if len(score_outliers) > 0:
        print(f"  High Outliers (> {Q3_score + 1.5 * IQR_score:.0f}): {len(score_outliers[score_outliers['score'] > Q3_score + 1.5 * IQR_score])}")
    
    # Comments outliers
    Q1_desc = df['descendants'].quantile(0.25)
    Q3_desc = df['descendants'].quantile(0.75)
    IQR_desc = Q3_desc - Q1_desc
    desc_outliers = df[(df['descendants'] < Q1_desc - 1.5 * IQR_desc) | 
                       (df['descendants'] > Q3_desc + 1.5 * IQR_desc)]
    
    print(f"\nComments Outliers: {len(desc_outliers)} ({len(desc_outliers)/len(df)*100:.2f}%)")
    if len(desc_outliers) > 0:
        top_comment_posts = desc_outliers.nlargest(5, 'descendants')[['title', 'descendants', 'score']]
        print(f"  Top 5 Most Commented Posts:")
        for idx, row in top_comment_posts.iterrows():
            print(f"    - {row['title'][:50]}... ({row['descendants']:.0f} comments)")

def save_analysis_report(results, filename="analysis_report.txt"):
    """
    Save analysis report to file
    
    Args:
        results (dict): Analysis results
        filename (str): Output filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("HACKERNEWS DATA ANALYSIS REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n✓ Analysis report saved to '{filename}'")
    except Exception as e:
        print(f"Error saving report: {e}")

def main():
    """Main execution function"""
    print("="*60)
    print("TASK 3: ANALYSE DATA USING NUMPY AND PANDAS")
    print("="*60)
    
    # Load data
    df = load_cleaned_data("hackernews_cleaned.csv")
    if df is None:
        return
    
    # Perform analysis
    results = statistical_analysis(df)
    
    # Additional analysis
    calculate_percentiles(df)
    identify_outliers(df)
    
    # Save report
    save_analysis_report(results)
    
    print("\n" + "="*60)
    print("Task 3 completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
