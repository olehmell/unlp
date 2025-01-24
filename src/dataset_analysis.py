import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def setup_ukrainian_plots():
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['figure.figsize'] = (12, 6)

def analyze_dataset(df):
    print("Basic Dataset Information:")
    print(f"Total number of posts: {len(df)}")
    
    # Manipulative distribution analysis
    print("\nManipulative vs Non-manipulative distribution:")
    manip_dist = df['manipulative'].value_counts()
    manip_percent = df['manipulative'].value_counts(normalize=True) * 100
    print(f"Manipulative: {manip_dist[True]} ({manip_percent[True]:.2f}%)")
    print(f"Non-manipulative: {manip_dist[False]} ({manip_percent[False]:.2f}%)")
    
    print(f"\nLanguage distribution:")
    print(df['lang'].value_counts(normalize=True) * 100)
    
    # Language and manipulative cross-analysis
    print("\nLanguage and Manipulative Cross-distribution:")
    lang_manip_dist = pd.crosstab(df['lang'], df['manipulative'], normalize='index') * 100
    print(lang_manip_dist)
    
    # Analyze techniques
    techniques = df[df['manipulative'] == True]['techniques']
    all_techniques = [tech for techs in techniques for tech in techs]
    technique_counts = Counter(all_techniques)
    
    # print("\nTechniques Distribution in Manipulative Posts:")
    for tech, count in technique_counts.most_common():
        percentage = (count / len(df[df['manipulative']]) * 100)
        print(f"{tech}: {count} ({percentage:.2f}%)")
    
    # Analyze number of techniques per manipulative post
    df['num_techniques'] = techniques.apply(len)
    print("\nTechniques per manipulative post statistics:")
    print(df[df['manipulative']]['num_techniques'].describe())
    
    # Analyze trigger words
    df['num_triggers'] = df['trigger_words'].fillna('[]').apply(len)
    print("\nTrigger words statistics (all posts):")
    print(df['num_triggers'].describe())
    
    print("\nTrigger words statistics (manipulative posts only):")
    print(df[df['manipulative']]['num_triggers'].describe())
    
    # Visualizations
    setup_ukrainian_plots()
    
    # 1. Manipulative vs Non-manipulative distribution
    plt.figure()
    sns.countplot(data=df, x='manipulative')
    plt.title('Distribution of Manipulative vs Non-manipulative Posts')
    plt.show()
    
    # 2. Language and Manipulative cross-distribution
    plt.figure()
    sns.heatmap(lang_manip_dist, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Language and Manipulative Cross-distribution (%)')
    plt.show()
    
    # 3. Techniques distribution in manipulative posts
    plt.figure(figsize=(15, 6))
    technique_df = pd.DataFrame.from_dict(technique_counts, orient='index', columns=['count'])
    sns.barplot(data=technique_df, x=technique_df.index, y='count')
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Techniques in Manipulative Posts')
    plt.tight_layout()
    plt.show()
    
    # 4. Number of techniques per manipulative post
    plt.figure()
    sns.histplot(data=df[df['manipulative']], x='num_techniques', 
                bins=range(max(df['num_techniques'])+2))
    plt.title('Distribution of Number of Techniques per Manipulative Post')
    plt.show()
    
    # 5. Correlation between number of techniques and trigger words (manipulative posts)
    plt.figure()
    sns.scatterplot(data=df[df['manipulative']], x='num_techniques', y='num_triggers')
    plt.title('Correlation: Number of Techniques vs Trigger Words (Manipulative Posts)')
    plt.show()
    
    # 6. Content length distribution by manipulative flag
    df['content_length'] = df['content'].str.len()
    plt.figure()
    sns.boxplot(data=df, x='manipulative', y='content_length')
    plt.title('Content Length Distribution by Manipulative Flag')
    plt.show()
    
    # 7. Language-specific technique usage (for manipulative posts)
    techniques_by_lang = {}
    for lang in df['lang'].unique():
        lang_techniques = [tech for techs in df[(df['lang'] == lang) & (df['manipulative'])]['techniques'] 
                         for tech in techs]
        techniques_by_lang[lang] = Counter(lang_techniques)
    
    lang_df = pd.DataFrame(techniques_by_lang)
    plt.figure(figsize=(15, 6))
    lang_df.plot(kind='bar')
    plt.title('Techniques Distribution by Language (Manipulative Posts)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # 8. Co-occurrence matrix (for manipulative posts)
    technique_list = sorted(set(all_techniques))
    cooccurrence = np.zeros((len(technique_list), len(technique_list)))
    
    for techs in df[df['manipulative']]['techniques']:
        for i, tech1 in enumerate(technique_list):
            for j, tech2 in enumerate(technique_list):
                if tech1 in techs and tech2 in techs:
                    cooccurrence[i][j] += 1
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cooccurrence, 
                xticklabels=technique_list, 
                yticklabels=technique_list,
                cmap='YlOrRd')
    plt.title('Technique Co-occurrence Matrix (Manipulative Posts)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 9. Additional statistical tests
    # Chi-square test for language and manipulative relationship
    from scipy.stats import chi2_contingency
    chi2, p_value, dof, expected = chi2_contingency(pd.crosstab(df['lang'], df['manipulative']))
    print("\nStatistical Tests:")
    print(f"Chi-square test for language and manipulative relationship: p-value = {p_value:.4f}")
    
    # T-test for content length between manipulative and non-manipulative posts
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(
        df[df['manipulative']]['content_length'],
        df[~df['manipulative']]['content_length']
    )
    print(f"T-test for content length difference: p-value = {p_value:.4f}")


# Read the parquet file
df = pd.read_parquet('data/bin/train.parquet')  # or however your data is stored
analyze_dataset(df)