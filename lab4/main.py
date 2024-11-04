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
    data = preprocessData(data_file)

    # Exploratory Data Analysis (EDA)
    statistics = data[['Benefits', 'TotalPay', 'TotalPayBenefits', 'Year']].describe()
    statistics_df = pd.DataFrame(statistics)
    statistics_str = tabulate(statistics_df, headers='keys', tablefmt='grid', stralign='center', numalign='center')

    # Save the statistics to a text file
    with open(f"{results_dir}/statistics.txt", "w") as file:
        file.write(statistics_str)

    # Distribution of BasePay
    plt.figure(figsize=(10, 6))
    sns.histplot(data['BasePay'].dropna(), kde=True, color='green', bins=30)
    plt.title("Distribution of BasePay")
    plt.xlabel("BasePay")
    plt.ylabel("Frequency")
    plt.savefig(f"{results_dir}/basepay_histogram.png")
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
    trainColumns = ['TotalPay', 'Benefits']
    predictedData, elasticNetModel = trainModels(data, results_dir, trainColumns)

    # Train with one necessary column removed
    trainColumns = ['TotalPay']
    trainModels(data, results_dir, trainColumns, remove=True)

    # Cluster and Visualize
    clusterAndVisualize(data, results_dir)

    # Analyze best model predictions with clusters
    predictedClusters(predictedData, results_dir, elasticNetModel)