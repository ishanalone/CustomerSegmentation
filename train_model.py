import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import os

def clean_data(df):
    """Clean and preprocess the raw data"""
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Convert date columns to datetime
    date_columns = ['Order Date / Time', 'Due Date', 'Last Activity', 'Last Payment Activity']
    for col in date_columns:
        try:
            data[col] = pd.to_datetime(data[col], dayfirst=True, errors='coerce')
        except:
            print(f"Warning: Could not convert all values in {col} to datetime")
    
    # Clean numeric columns
    numeric_columns = ['Pcs.', 'Weight', 'Gross Amount', 'Discount', 'Tax', 
                      'Net Amount', 'Advance', 'Paid', 'Adjustment', 'Balance']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Fill missing values
    data['Weight'] = data['Weight'].fillna(0)
    data['Discount'] = data['Discount'].fillna(0)
    data['Package'] = data['Package'].fillna('No')
    data['Primary Services'] = data['Primary Services'].fillna('Unknown')
    
    # Add service type based on weight
    data['Service_Type'] = np.where(data['Weight'] > 0, 'Laundry', 'Dry Clean')
    
    # Remove rows with critical missing values
    data = data.dropna(subset=['Customer Code', 'Net Amount', 'Order Date / Time'])
    
    return data

def calculate_customer_metrics(data):
    """Calculate customer-level metrics with service type breakdown"""
    # Group by Customer Code to get customer-level metrics
    customer_metrics = data.groupby('Customer Code').agg({
        'Order No.': 'count',                    # Total orders
        'Net Amount': ['sum', 'mean'],           # Total and average spending
        'Last Activity': 'max',                  # Last order date
        'Pcs.': ['sum', 'mean'],                # Total and average pieces
        'Weight': ['sum', 'mean'],              # Total and average weight
        'Discount': ['mean', 'sum'],            # Average and total discount
        'Primary Services': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown',
        'Order Status': lambda x: (x == 'Delivered').mean(),
        'Package': lambda x: x.eq('Yes').mean()
    }).reset_index()

    # Flatten the multi-level columns first
    customer_metrics.columns = [
        'Customer_Code',
        'Order_Frequency',
        'Total_Spending', 'Avg_Order_Value',
        'Last_Activity',
        'Total_Pieces', 'Avg_Pieces_Per_Order',
        'Total_Weight', 'Avg_Weight_Per_Order',
        'Avg_Discount_Rate', 'Total_Discount',
        'Primary_Service',
        'Delivery_Rate',
        'Package_Usage_Rate'
    ]

    # Calculate service type specific metrics
    laundry_metrics = data[data['Service_Type'] == 'Laundry'].groupby('Customer Code').agg({
        'Order No.': 'count',
        'Net Amount': 'sum',
        'Weight': 'sum'
    }).reset_index()
    laundry_metrics.columns = ['Customer_Code', 'Laundry_Orders', 'Laundry_Spending', 'Laundry_Weight']

    dryclean_metrics = data[data['Service_Type'] == 'Dry Clean'].groupby('Customer Code').agg({
        'Order No.': 'count',
        'Net Amount': 'sum',
        'Pcs.': 'sum'
    }).reset_index()
    dryclean_metrics.columns = ['Customer_Code', 'DryClean_Orders', 'DryClean_Spending', 'DryClean_Pieces']

    # Merge all metrics
    customer_metrics = pd.merge(customer_metrics, laundry_metrics, 
                              left_on='Customer_Code', right_on='Customer_Code', how='left')
    customer_metrics = pd.merge(customer_metrics, dryclean_metrics, 
                              left_on='Customer_Code', right_on='Customer_Code', how='left')

    # Fill NaN values with 0 for service-specific metrics
    service_columns = ['Laundry_Orders', 'Laundry_Spending', 'Laundry_Weight',
                      'DryClean_Orders', 'DryClean_Spending', 'DryClean_Pieces']
    customer_metrics[service_columns] = customer_metrics[service_columns].fillna(0)

    # Calculate service preference ratio
    customer_metrics['Laundry_Ratio'] = customer_metrics['Laundry_Orders'] / customer_metrics['Order_Frequency']
    customer_metrics['DryClean_Ratio'] = customer_metrics['DryClean_Orders'] / customer_metrics['Order_Frequency']

    return customer_metrics

def train_model():
    # Load and clean data
    raw_data = pd.read_csv('311224.csv', encoding='windows-1252')
    data = clean_data(raw_data)
    
    # Calculate customer metrics
    customer_metrics = calculate_customer_metrics(data)
    
    # Calculate recency
    customer_metrics['Last_Activity'] = pd.to_datetime(customer_metrics['Last_Activity'])
    latest_date = customer_metrics['Last_Activity'].max()
    customer_metrics['Recency'] = (latest_date - customer_metrics['Last_Activity']).dt.days
    
    # Calculate additional metrics
    customer_metrics['Avg_Monthly_Orders'] = customer_metrics['Order_Frequency'] / 12
    customer_metrics['Value_Per_Kg'] = np.where(
        customer_metrics['Total_Weight'] > 0,
        customer_metrics['Laundry_Spending'] / customer_metrics['Total_Weight'],
        0
    )
    
    # Select features for clustering
    features_for_clustering = [
        'Order_Frequency',
        'Total_Spending',
        'Avg_Order_Value',
        'Recency',
        'Delivery_Rate',
        'Package_Usage_Rate',
        'Laundry_Ratio',
        'DryClean_Ratio',
        'Value_Per_Kg',
        'Avg_Monthly_Orders'
    ]
    
    # Prepare data for clustering
    X = customer_metrics[features_for_clustering].copy()
    X = X.fillna(X.mean())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KMeans model
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)
    
    # Add segments to customer_metrics
    customer_metrics['Segment'] = kmeans.predict(X_scaled)
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Save files to static directory
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=features_for_clustering
    )
    cluster_centers.to_csv('static/cluster_characteristics.csv', index=False)
    customer_metrics.to_csv('static/customer_segments.csv', index=False)
    
    # Save model files
    joblib.dump(kmeans, 'static/customer_segmentation_model.pkl')
    joblib.dump(scaler, 'static/scaler.pkl')
    
    # Print summary statistics
    print("\nModel training completed successfully!")
    print(f"Total customers processed: {len(customer_metrics)}")
    print("\nSegment distribution:")
    print(customer_metrics['Segment'].value_counts())
    print("\nService Type Distribution:")
    print(f"Average Laundry Ratio: {customer_metrics['Laundry_Ratio'].mean():.2%}")
    print(f"Average Dry Clean Ratio: {customer_metrics['DryClean_Ratio'].mean():.2%}")
    
    # Print segment characteristics
    print("\nSegment Characteristics:")
    segment_summary = customer_metrics.groupby('Segment').agg({
        'Laundry_Ratio': 'mean',
        'DryClean_Ratio': 'mean',
        'Total_Spending': 'mean',
        'Order_Frequency': 'mean'
    }).round(2)
    print(segment_summary)
    
    return customer_metrics  # Optional: return the metrics for further analysis

if __name__ == "__main__":
    train_model() 