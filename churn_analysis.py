import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
import os

def identify_churn_risk(data):
    """
    Identify customers at risk of churning based on:
    1. Recency of last order
    2. Change in order frequency
    3. Change in spending patterns
    4. Decline in engagement metrics
    """
    
    # Calculate recency thresholds (in days)
    CHURNED_THRESHOLD = 180  # 6 months
    HIGH_RISK_THRESHOLD = 90  # 3 months
    MEDIUM_RISK_THRESHOLD = 60  # 2 months
    
    # Convert date columns to datetime
    date_columns = ['Order Date / Time', 'Due Date', 'Last Activity', 'Last Payment Activity']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], dayfirst=True, errors='coerce')
    
    # Print column names to debug
    print("Available columns:", data.columns.tolist())
    
    # Calculate metrics for each customer
    customer_metrics = []
    current_date = datetime.now()
    
    for customer_code, customer_data in data.groupby('Customer Code'):
        try:
            # Sort orders by date and remove rows with NaT dates
            customer_data = customer_data.dropna(subset=['Order Date / Time'])
            if len(customer_data) == 0:
                print(f"Skipping customer {customer_code}: No valid order dates")
                continue
                
            customer_data = customer_data.sort_values('Order Date / Time')
            
            # Get customer name and mobile (check multiple possible column names)
            customer_name = None
            for name_col in ['Customer Name', 'Customer_Name', 'CustomerName', 'Name', 'Customer']:
                if name_col in customer_data.columns:
                    customer_name = customer_data.iloc[-1][name_col]
                    if pd.notnull(customer_name) and str(customer_name).strip():
                        break
            customer_name = customer_name if pd.notnull(customer_name) and str(customer_name).strip() else 'Unknown'
            
            # Get mobile number (check multiple possible column names)
            customer_mobile = None
            for mobile_col in ['Mobile No', 'Mobile_No', 'Mobile', 'Phone', 'Contact']:
                if mobile_col in customer_data.columns:
                    customer_mobile = customer_data.iloc[-1][mobile_col]
                    if pd.notnull(customer_mobile) and str(customer_mobile).strip():
                        break
            customer_mobile = customer_mobile if pd.notnull(customer_mobile) and str(customer_mobile).strip() else 'Unknown'
            
            # Print debug info for first few customers
            if len(customer_metrics) < 5:
                print(f"Debug - Customer {customer_code}:")
                print(f"Name: {customer_name}")
                print(f"Mobile: {customer_mobile}")
                print("Available columns:", customer_data.columns.tolist())
            
            # Basic metrics
            total_orders = len(customer_data)
            total_spending = customer_data['Net Amount'].sum()
            
            # Ensure dates are datetime objects and handle NaT
            last_order_date = customer_data['Order Date / Time'].max()
            first_order_date = customer_data['Order Date / Time'].min()
            
            if pd.isna(last_order_date) or pd.isna(first_order_date):
                print(f"Skipping customer {customer_code}: Invalid order dates")
                continue
            
            # Calculate days since last order
            days_since_last_order = (current_date - last_order_date).days
            
            # Calculate frequency changes
            if total_orders >= 2:
                # Split orders into two periods
                mid_point = first_order_date + (last_order_date - first_order_date) / 2
                recent_orders = customer_data[customer_data['Order Date / Time'] > mid_point]
                older_orders = customer_data[customer_data['Order Date / Time'] <= mid_point]
                
                # Calculate time periods in days and convert to months
                recent_period = max((last_order_date - mid_point).days / 30, 1)
                older_period = max((mid_point - first_order_date).days / 30, 1)
                
                # Calculate frequency change
                recent_frequency = len(recent_orders) / recent_period
                older_frequency = len(older_orders) / older_period
                frequency_change = (recent_frequency - older_frequency) / (older_frequency if older_frequency > 0 else 1)
                
                # Calculate spending change
                recent_avg_spending = recent_orders['Net Amount'].mean()
                older_avg_spending = older_orders['Net Amount'].mean()
                spending_change = (recent_avg_spending - older_avg_spending) / (older_avg_spending if older_avg_spending > 0 else 1)
            else:
                frequency_change = 0
                spending_change = 0
            
            # Determine churn risk
            if days_since_last_order > CHURNED_THRESHOLD:
                churn_status = 'Churned'
                risk_score = 1.0
            else:
                # Calculate risk score (0 to 1)
                recency_score = min(days_since_last_order / CHURNED_THRESHOLD, 1)
                frequency_score = max(min(-frequency_change, 1), 0) if frequency_change < 0 else 0
                spending_score = max(min(-spending_change, 1), 0) if spending_change < 0 else 0
                
                risk_score = (recency_score * 0.4 + frequency_score * 0.3 + spending_score * 0.3)
                
                if days_since_last_order > HIGH_RISK_THRESHOLD:
                    churn_status = 'High Risk'
                elif days_since_last_order > MEDIUM_RISK_THRESHOLD:
                    churn_status = 'Medium Risk'
                else:
                    churn_status = 'Low Risk'
            
            # Format dates safely
            try:
                last_order_str = last_order_date.strftime('%Y-%m-%d')
                first_order_str = first_order_date.strftime('%Y-%m-%d')
            except:
                last_order_str = str(last_order_date)[:10]
                first_order_str = str(first_order_date)[:10]
            
            customer_metrics.append({
                'Customer_Code': customer_code,
                'Customer_Name': str(customer_name).strip(),  # Clean up the name
                'Mobile_No': str(customer_mobile).strip(),    # Clean up the number
                'Total_Orders': total_orders,
                'Total_Spending': round(total_spending, 2),
                'Last_Order_Date': last_order_str,
                'First_Order_Date': first_order_str,
                'Days_Since_Last_Order': days_since_last_order,
                'Frequency_Change': round(frequency_change, 2),
                'Spending_Change': round(spending_change, 2),
                'Risk_Score': round(risk_score, 2),
                'Churn_Status': churn_status,
                'Monthly_Frequency': round(total_orders / max(((last_order_date - first_order_date).days / 30), 1), 2),
                'Avg_Spending': round(total_spending / total_orders, 2) if total_orders > 0 else 0
            })
            
        except Exception as e:
            print(f"Error processing customer {customer_code}: {str(e)}")
            continue
    
    # Create DataFrame and clean up names/numbers
    df = pd.DataFrame(customer_metrics)
    
    # Clean up customer names
    df['Customer_Name'] = df['Customer_Name'].apply(lambda x: x if pd.notnull(x) and str(x).strip() != '' else 'Unknown')
    df['Mobile_No'] = df['Mobile_No'].apply(lambda x: x if pd.notnull(x) and str(x).strip() != '' else 'Unknown')
    
    return df

def generate_churn_insights(churn_data):
    """Generate insights and recommendations for at-risk customers"""
    
    def get_recommendations(row):
        recommendations = []
        
        # Add contact info to recommendations
        contact_info = f"Contact: {row['Customer_Name']} ({row['Mobile_No']})"
        recommendations.append(contact_info)
        
        if row['Churn_Status'] == 'Churned':
            recommendations.extend([
                "Send win-back email with special discount",
                "Personal call from customer service",
                "Offer free pickup and delivery for next order"
            ])
        elif row['Churn_Status'] == 'High Risk':
            recommendations.extend([
                "Send retention offer",
                "Request feedback on last service",
                "Promote new services or packages"
            ])
        elif row['Churn_Status'] == 'Medium Risk':
            recommendations.extend([
                "Send engagement email",
                "Offer loyalty program benefits",
                "Share seasonal promotions"
            ])
        
        if row['Frequency_Change'] < -0.2:
            recommendations.append(f"Address {abs(row['Frequency_Change']*100):.0f}% drop in visit frequency")
        if row['Spending_Change'] < -0.2:
            recommendations.append(f"Investigate {abs(row['Spending_Change']*100):.0f}% decrease in spending")
        
        return '; '.join(recommendations)
    
    churn_data['Recommendations'] = churn_data.apply(get_recommendations, axis=1)
    
    return churn_data

def create_churn_visualizations(churn_data):
    """Create visualizations for churn analysis"""
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Ensure required columns exist
    required_columns = ['Customer_Name', 'Mobile_No']
    for col in required_columns:
        if col not in churn_data.columns:
            churn_data[col] = 'Unknown'
    
    # Create risk distribution pie chart
    fig_risk = px.pie(
        churn_data, 
        names='Churn_Status',
        title='Customer Churn Risk Distribution'
    )
    fig_risk.write_html('static/churn_risk_distribution.html')
    
    # Create scatter plot of recency vs risk score
    hover_data = ['Total_Orders', 'Total_Spending']
    if 'Customer_Name' in churn_data.columns:
        hover_data.append('Customer_Name')
    if 'Mobile_No' in churn_data.columns:
        hover_data.append('Mobile_No')
        
    fig_scatter = px.scatter(
        churn_data,
        x='Days_Since_Last_Order',
        y='Risk_Score',
        color='Churn_Status',
        hover_data=hover_data,
        title='Customer Risk Analysis',
        labels={
            'Days_Since_Last_Order': 'Days Since Last Order',
            'Risk_Score': 'Risk Score'
        }
    )
    fig_scatter.write_html('static/churn_risk_scatter.html')
    
    # Create summary table with more detailed information
    summary_html = f"""
    <div class="churn-summary">
        <h3>Churn Risk Summary</h3>
        <table class="table table-striped">
            <tr>
                <th>Status</th>
                <th>Count</th>
                <th>Percentage</th>
                <th>Avg Days Since Order</th>
                <th>Avg Spending</th>
            </tr>
            {generate_summary_rows(churn_data)}
        </table>
    </div>
    """
    
    with open('static/churn_summary.html', 'w') as f:
        f.write(summary_html)

def generate_summary_rows(churn_data):
    """Generate HTML rows for churn summary table"""
    total = len(churn_data)
    rows = []
    
    for status in ['Churned', 'High Risk', 'Medium Risk', 'Low Risk']:
        status_data = churn_data[churn_data['Churn_Status'] == status]
        count = len(status_data)
        percentage = (count / total) * 100
        avg_days = status_data['Days_Since_Last_Order'].mean()
        avg_spending = status_data['Total_Spending'].mean()
        
        row = f"""
        <tr>
            <td>{status}</td>
            <td>{count}</td>
            <td>{percentage:.1f}%</td>
            <td>{avg_days:.1f}</td>
            <td>₹{avg_spending:,.2f}</td>
        </tr>
        """
        rows.append(row)
    
    return '\n'.join(rows)

def main():
    print("Starting churn analysis...")
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    try:
        raw_data = pd.read_csv('311224.csv', encoding='windows-1252')
        print(f"Loaded {len(raw_data)} records")
        
        # Print column information
        print("\nColumns in raw data:")
        for col in raw_data.columns:
            print(f"Column: {col}")
            print(f"Sample values: {raw_data[col].head().tolist()}")
            print(f"Null count: {raw_data[col].isnull().sum()}")
            print("---")
        
        # Perform churn analysis
        churn_data = identify_churn_risk(raw_data)
        
        # Verify the results
        print("\nProcessed data info:")
        print(churn_data.info())
        print("\nSample of processed customers:")
        print(churn_data[['Customer_Code', 'Customer_Name', 'Mobile_No']].head())
        
        # Generate insights
        churn_insights = generate_churn_insights(churn_data)
        
        # Save results
        churn_insights.to_csv('static/churn_analysis.csv', index=False)
        
        # Create visualizations
        create_churn_visualizations(churn_insights)
        
        print("\nChurn analysis complete!")
        print(f"\nChurn Risk Distribution:")
        print(churn_insights['Churn_Status'].value_counts())
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise  # Re-raise the exception to see the full traceback


if __name__ == "__main__":
    main() 