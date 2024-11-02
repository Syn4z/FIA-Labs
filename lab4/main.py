import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
    pay_columns = ['BasePay', 'OvertimePay', 'OtherPay', 'Benefits', 'TotalPay']
    data[pay_columns] = data[pay_columns].apply(pd.to_numeric, errors='coerce')

    # Calculate TotalPayBenefits
    data['TotalPayBenefits'] = data['TotalPay'] + data['Benefits']

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

    # Top 10 Job Titles by Average TotalPayBenefits
    top_jobs = data.groupby('JobTitle')['TotalPayBenefits'].mean().nlargest(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_jobs.values, y=top_jobs.index, hue=top_jobs.index, palette='plasma', dodge=False, legend=False)
    plt.title("Top 10 Job Titles by Average TotalPayBenefits")
    plt.xlabel("Average TotalPayBenefits")
    plt.ylabel("Job Title")
    plt.savefig(f"{results_dir}/top_job_titles.png", bbox_inches='tight')
    plt.close()

    # Step 4: Prepare the features and target variable
    X = data[['TotalPay', 'Benefits']]
    y = data['TotalPayBenefits']  # Updated target variable

    # Step 5: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6: Initialize and fit the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 7: Make predictions
    y_pred = model.predict(X_test)

    # Step 8: Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"R-squared: {r2}")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

    # Step 9: Visualize predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, label='Predicted', color='blue', alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Actual')  
    plt.title('Actual vs Predicted TotalPayBenefits')
    plt.legend()
    plt.savefig(f"{results_dir}/actual_vs_predicted.png")
    plt.close()

