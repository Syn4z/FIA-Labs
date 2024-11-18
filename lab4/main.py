import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os
from utils import *

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
data_file = "data.csv"

if __name__ == '__main__':
    data = preprocessData(data_file, results_dir)

    float_columns = data.select_dtypes(include=['float64']).columns
    float_data = data[float_columns]
    # Generate correlation matrix
    correlation_matrix = float_data.corr()

    # Plot the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig(f"{results_dir}/correlation_matrix.png")
    plt.close()

    # Exploratory Data Analysis (EDA)
    statistics = data[['Benefits', 'TotalPay', 'BasePay', 'OtherPay']].describe()
    statistics_df = pd.DataFrame(statistics)
    statistics_str = tabulate(statistics_df, headers='keys', tablefmt='grid', stralign='center', numalign='center')

    # Save the statistics to a text file
    with open(f"{results_dir}/statistics.txt", "w") as file:
        file.write(statistics_str)

    # Subplot for BasePay
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data['BasePay'].dropna(), kde=True, color='green', bins=30)
    plt.title("Distribution of BasePay")
    plt.xlabel("BasePay")
    plt.ylabel("Frequency")

    # Subplot for Benefits
    plt.subplot(1, 2, 2)
    sns.histplot(data['Benefits'].dropna(), kde=True, color='blue', bins=30)
    plt.title("Distribution of Benefits")
    plt.xlabel("Benefits")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f"{results_dir}/basepay_benefits_histogram.png", bbox_inches='tight')
    plt.close()

    # Top 10 Job Titles by Average TotalPayBenefits
    top_jobs = data.groupby('JobTitle')['TotalPayBenefits'].mean().nlargest(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_jobs.values, y=top_jobs.index, hue=top_jobs.index, palette='plasma', dodge=False, legend=False)
    plt.title("Top 10 Job Titles by Average TotalPayBenefits")
    plt.xlabel("Average TotalPayBenefits")
    plt.ylabel("Job Title")
    plt.savefig(f"{results_dir}/top_job_titles.png", bbox_inches='tight')
    plt.close()

    # Train and Evaluate Models
    trainColumns = ['BasePay', 'OtherPay']
    targetColumn = 'TotalPay'
    predictedData, lars_model = trainModels(data, results_dir, trainColumns, targetColumn)

    # Cluster and Visualize
    clusterAndVisualize(data, results_dir)

    # Analyze best model predictions with clusters
    predictedClusters(predictedData, results_dir, lars_model)