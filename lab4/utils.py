from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, Lars
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


def preprocessData(filename):
    data = pd.read_csv(
        filename,
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
    return data

def trainModels(data, results_dir, trainColumns, remove=False):
    # Select the features and target variable
    X = data[trainColumns]
    y = data['TotalPayBenefits']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear_model = LinearRegression()
    elastic_net_model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
    lars_model = Lars()

    # Train models
    linear_model.fit(X_train, y_train)
    elastic_net_model.fit(X_train, y_train)
    lars_model.fit(X_train, y_train)

    # Make predictions
    y_pred_linear = linear_model.predict(X_test)
    y_pred_elastic = elastic_net_model.predict(X_test)
    y_pred_lars = lars_model.predict(X_test)

    # Evaluate models
    metrics = {
        "Model": ["Linear Regression", "Elastic Net Regularization", "Least-angle Regression (LARS)"],
        "Mean Squared Error": [
            mean_squared_error(y_test, y_pred_linear),
            mean_squared_error(y_test, y_pred_elastic),
            mean_squared_error(y_test, y_pred_lars)
        ],
        "Mean Absolute Error": [
            mean_absolute_error(y_test, y_pred_linear),
            mean_absolute_error(y_test, y_pred_elastic),
            mean_absolute_error(y_test, y_pred_lars)
        ]
    }
    
    metrics_df = pd.DataFrame(metrics)
    metrics_str = tabulate(metrics_df, headers='keys', tablefmt='grid', showindex=False, stralign='center', numalign='center')
    
    if remove:
        with open(f"{results_dir}/model_metrics_remove.txt", "w") as file:
            file.write(metrics_str)
    else:
        with open(f"{results_dir}/model_metrics.txt", "w") as file:
            file.write(metrics_str)        

    models = {
        "Linear Regression": (y_pred_linear, 'blue'),
        "Elastic Net Regularization": (y_pred_elastic, 'green'),
        "LARS": (y_pred_lars, 'purple')
    }

    # Visualize the results
    for model_name, (predictions, color) in models.items():
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, label='Predicted', color=color, alpha=0.6)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Actual')
        plt.title(f'Actual vs Predicted TotalPayBenefits ({model_name})')
        plt.xlabel("Actual TotalPayBenefits")
        plt.ylabel("Predicted TotalPayBenefits")
        plt.legend()
        if remove:
            plt.savefig(f"{results_dir}/{model_name.lower().replace(' ', '_')}_remove.png")
        else:
            plt.savefig(f"{results_dir}/{model_name.lower().replace(' ', '_')}.png")
        plt.close()

    return data, elastic_net_model

def clusterAndVisualize(data, results_dir):
    # Select relevant features for clustering
    features = data[['TotalPay', 'Benefits']]
    # Scale the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    # Fit the K-means model
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(scaled_features)
    # Assign clusters to the original data
    data['Cluster'] = kmeans.labels_
    # Perform PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
   
    # Define a fixed color map
    color_map = {0: 'blueviolet', 1: 'seagreen', 2: 'deepskyblue'}
    colors = list(color_map.values())

    plt.figure(figsize=(10, 12))
    plt.subplot(2, 1, 1)
    scatter = plt.scatter(data['TotalPay'], data['Benefits'], 
                          c=data['Cluster'].map(color_map), alpha=0.5)
    plt.title('K-means Clustering')
    plt.xlabel('Benefits')
    plt.ylabel('TotalPayBenefits')
    plt.grid()
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=color, markersize=10) for color in colors]
    plt.legend(handles, [f'Cluster {i + 1}' for i in range(len(colors))], title='Clusters')

    # Subplot for PCA-transformed data
    plt.subplot(2, 1, 2)
    for cluster, color in color_map.items():
        plt.scatter(pca_features[data['Cluster'] == cluster, 0], 
                    pca_features[data['Cluster'] == cluster, 1], 
                    s=100, label=f'Cluster {cluster + 1}', alpha=0.6,
                    color=color)
                   
    # Plotting the cluster centers
    centers = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', label='Centroids', marker='X')
    plt.title('K-means Clustering of TotalPayBenefits (PCA Projection)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/cluster_visualization.png")
    plt.close()

def predictedClusters(data, results_dir, elastic_net_model):
    features = data[['TotalPay', 'Benefits']]
    # Scale the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    # Fit the K-means model
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(scaled_features)

    data['Cluster'] = kmeans.labels_
    data['PredictedTotalPayBenefits'] = elastic_net_model.predict(data[['TotalPay', 'Benefits']])

    # Define colors for clusters
    color_map = {0: 'blue', 1: 'green', 2: 'orange'}
    plt.figure(figsize=(10, 8))
    
    # Plot each cluster
    for cluster, color in color_map.items():
        plt.scatter(data[data['Cluster'] == cluster]['TotalPay'], 
                    data[data['Cluster'] == cluster]['PredictedTotalPayBenefits'], 
                    c=color, 
                    label=f'Cluster {cluster}', 
                    alpha=0.7)

    max_value = max(data['TotalPayBenefits'].max(), data['PredictedTotalPayBenefits'].max())
    plt.plot([0, max_value], [0, max_value], color='red', linestyle='--', linewidth=2, label='Actual Values')
    plt.title('Predicted (Elastic Net Regularization) vs Actual TotalPayBenefits by Clusters')
    plt.xlabel('TotalPayBenefits')
    plt.ylabel('Predicted TotalPayBenefits')
    plt.legend()
    plt.savefig(f"{results_dir}/predicted_vs_actual_clusters.png")
    plt.close()
