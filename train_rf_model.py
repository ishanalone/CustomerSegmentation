import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  # For initial labels
import joblib
from datetime import datetime
import os

def clean_data(df):
    """Clean and preprocess the raw data"""
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
        'Order No.': 'count',
        'Net Amount': ['sum', 'mean'],
        'Last Activity': 'max',
        'Pcs.': ['sum', 'mean'],
        'Weight': ['sum', 'mean'],
        'Discount': ['mean', 'sum'],
        'Primary Services': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown',
        'Order Status': lambda x: (x == 'Delivered').mean(),
        'Package': lambda x: x.eq('Yes').mean()
    }).reset_index()

    # Calculate service type specific metrics
    laundry_metrics = data[data['Service_Type'] == 'Laundry'].groupby('Customer Code').agg({
        'Order No.': 'count',
        'Net Amount': 'sum',
        'Weight': 'sum'
    }).reset_index()
    
    dryclean_metrics = data[data['Service_Type'] == 'Dry Clean'].groupby('Customer Code').agg({
        'Order No.': 'count',
        'Net Amount': 'sum',
        'Pcs.': 'sum'
    }).reset_index()

    # Merge all metrics
    customer_metrics.columns = [
        'Customer_Code', 'Order_Frequency', 'Total_Spending', 'Avg_Order_Value',
        'Last_Activity', 'Total_Pieces', 'Avg_Pieces_Per_Order', 'Total_Weight',
        'Avg_Weight_Per_Order', 'Avg_Discount_Rate', 'Total_Discount',
        'Primary_Service', 'Delivery_Rate', 'Package_Usage_Rate'
    ]
    
    laundry_metrics.columns = ['Customer_Code', 'Laundry_Orders', 'Laundry_Spending', 'Laundry_Weight']
    dryclean_metrics.columns = ['Customer_Code', 'DryClean_Orders', 'DryClean_Spending', 'DryClean_Pieces']

    # Merge metrics
    customer_metrics = pd.merge(customer_metrics, laundry_metrics, how='left', on='Customer_Code')
    customer_metrics = pd.merge(customer_metrics, dryclean_metrics, how='left', on='Customer_Code')

    # Fill NaN values
    service_columns = ['Laundry_Orders', 'Laundry_Spending', 'Laundry_Weight',
                      'DryClean_Orders', 'DryClean_Spending', 'DryClean_Pieces']
    customer_metrics[service_columns] = customer_metrics[service_columns].fillna(0)

    # Calculate service preference ratios
    customer_metrics['Laundry_Ratio'] = customer_metrics['Laundry_Orders'] / customer_metrics['Order_Frequency']
    customer_metrics['DryClean_Ratio'] = customer_metrics['DryClean_Orders'] / customer_metrics['Order_Frequency']

    return customer_metrics

def train_rf_model():
    """Train Random Forest model for customer segmentation"""
    print("Starting Random Forest model training...")
    
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
    features = [
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
    
    # Prepare data
    X = customer_metrics[features].copy()
    X = X.fillna(X.mean())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # First use KMeans to get initial labels
    kmeans = KMeans(n_clusters=3, random_state=42)
    initial_labels = kmeans.fit_predict(X_scaled)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_scaled, initial_labels)
    
    # Get feature importances
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Save models and data
    joblib.dump(rf_model, 'static/rf_segmentation_model.pkl')
    joblib.dump(scaler, 'static/rf_scaler.pkl')
    
    # Add segments to customer metrics
    customer_metrics['RF_Segment'] = rf_model.predict(X_scaled)
    customer_metrics.to_csv('static/rf_customer_segments.csv', index=False)
    
    # Save feature importance
    feature_importance.to_csv('static/rf_feature_importance.csv', index=False)
    
    # Print summary statistics
    print("\nRandom Forest model training completed!")
    print(f"Total customers processed: {len(customer_metrics)}")
    print("\nSegment distribution:")
    print(customer_metrics['RF_Segment'].value_counts())
    print("\nTop 5 most important features:")
    print(feature_importance.head())
    
    return customer_metrics, feature_importance

if __name__ == "__main__":
    train_rf_model() 