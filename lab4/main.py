import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os


os.makedirs("results", exist_ok=True)
results_dir = "results"

if __name__ == '__main__':
    # Load the dataset
    data = pd.read_csv(
    "data.csv",
    dtype={
        'Id': int,
        'EmployeeName': str,
        'JobTitle': str,
        'BasePay': float,
        'OvertimePay': float,
        'OtherPay': float,
        'Benefits': float,
        'TotalPay': float,
        'TotalPayBenefits': float,
        'Year': int,
        'Notes': str,
        'Agency': str,
        'Status': str
    },
    na_values=["Not Provided", "N/A", "", " "]
)

    # 1. Data Preprocessing
    # Check for missing values
    print("Missing values per column:\n", data.isnull().sum())

    # Fill missing values in Benefits and Status (or drop if not needed)
    data['Benefits'] = data['Benefits'].fillna(0)
    data['Status'] = data['Status'].fillna("Unknown")

    # Convert columns to appropriate data types
    pay_columns = ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits', 'TotalPay', 'TotalPayBenefits']
    data[pay_columns] = data[pay_columns].apply(pd.to_numeric, errors='coerce')

    # Remove duplicates based on Id
    data.drop_duplicates(subset="Id", inplace=True)

    # 2. Exploratory Data Analysis (EDA)
    statistics = data[['BasePay', 'TotalPay', 'TotalPayBenefits', 'Year']].describe()
    statistics_df = pd.DataFrame(statistics)
    statistics_str = tabulate(statistics_df, headers='keys', tablefmt='grid', stralign='center', numalign='center')

    # Save the statistics to a text file
    with open(f"{results_dir}/statistics.txt", "w") as file:
        file.write(statistics_str)

    # 3. Data Visualization
    # Distribution of BasePay and OvertimePay
    plt.figure(figsize=(10, 6))
    sns.histplot(data['BasePay'].dropna(), kde=True, color='green', bins=30)
    plt.title("Distribution of BasePay")
    plt.xlabel("BasePay")
    plt.ylabel("Frequency")
    plt.savefig(f"{results_dir}/basepay_histogram.png")
    plt.close()

    # Top 10 Job Titles by Average TotalPay
    top_jobs = data.groupby('JobTitle')['TotalPay'].mean().nlargest(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_jobs.values, y=top_jobs.index, hue=top_jobs.index, palette='plasma', dodge=False, legend=False)
    plt.title("Top 10 Job Titles by Average Total Pay")
    plt.xlabel("Average Total Pay")
    plt.ylabel("Job Title")
    plt.savefig(f"{results_dir}/top_job_titles.png", bbox_inches='tight')
    plt.close()
