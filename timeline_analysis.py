import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

def create_timeline_statistics(data):
    """
    Create timeline statistics and visualizations:
    1. Monthly order trends
    2. Revenue trends
    3. Customer acquisition timeline
    4. Service preference changes over time
    5. Average order value trends
    """
    
    # Filter out orders with 'A' prefix
    data = data[~data['Order No.'].str.startswith('A', na=False)].copy()
    print(f"Filtered to {len(data)} records after removing 'A' prefix orders")
    
    # Find the date column (handle different possible names)
    date_column = None
    for col in ['Order Date / Time', 'Order Date/Time', 'Order_Date', 'OrderDate', 'Date']:
        if col in data.columns:
            date_column = col
            break
    
    if not date_column:
        raise ValueError("Could not find order date column in the data")
    
    print(f"Using date column: {date_column}")
    
    # Convert date column to datetime
    data[date_column] = pd.to_datetime(data[date_column], dayfirst=True)
    
    # Set the date as index
    data = data.set_index(date_column)
    
    # Sort by date
    data = data.sort_index()
    
    # Create monthly statistics
    monthly_stats = calculate_monthly_statistics(data)
    
    # Create visualizations
    create_timeline_visualizations(monthly_stats)
    
    return monthly_stats

def calculate_monthly_statistics(data):
    """Calculate various monthly statistics"""
    
    # Resample data by month (use 'ME' instead of 'M' to avoid warning)
    monthly_stats = pd.DataFrame()
    
    # Order counts
    monthly_stats['Order_Count'] = data.resample('ME').size()
    
    # Revenue
    monthly_stats['Revenue'] = data['Net Amount'].resample('ME').sum()
    
    # Unique customers
    monthly_stats['Unique_Customers'] = data['Customer Code'].resample('ME').nunique()
    
    # New customers (first time orders)
    # Get first order date for each customer
    customer_first_orders = data.reset_index().groupby('Customer Code')['Order Date / Time'].min()
    # Convert to DataFrame with datetime index
    first_orders_df = pd.DataFrame(customer_first_orders).set_index('Order Date / Time')
    # Count new customers per month
    monthly_stats['New_Customers'] = first_orders_df.resample('ME').size()
    
    # Average order value
    monthly_stats['Avg_Order_Value'] = monthly_stats['Revenue'] / monthly_stats['Order_Count']
    
    # Service type distribution
    service_column = 'Primary Services'  # Use Primary Services column
    if service_column in data.columns:
        # Split multiple services and explode
        services = data[service_column].str.split(',').explode().str.strip()
        service_types = pd.get_dummies(services)
        for service in service_types.columns:
            monthly_stats[f'Service_{service}'] = service_types[service].resample('ME').sum()
    
    # Calculate month-over-month growth rates
    monthly_stats['Revenue_Growth'] = monthly_stats['Revenue'].pct_change() * 100
    monthly_stats['Order_Growth'] = monthly_stats['Order_Count'].pct_change() * 100
    monthly_stats['Customer_Growth'] = monthly_stats['Unique_Customers'].pct_change() * 100
    
    return monthly_stats

def create_timeline_visualizations(monthly_stats):
    """Create timeline visualizations"""
    
    # 1. Order and Revenue Trends
    fig = go.Figure()
    
    # Add order count line
    fig.add_trace(go.Scatter(
        x=monthly_stats.index,
        y=monthly_stats['Order_Count'],
        name='Orders',
        line=dict(color='blue'),
        yaxis='y'
    ))
    
    # Add revenue line on secondary axis
    fig.add_trace(go.Scatter(
        x=monthly_stats.index,
        y=monthly_stats['Revenue'],
        name='Revenue',
        line=dict(color='green'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Monthly Orders and Revenue Trends',
        yaxis=dict(title='Number of Orders', side='left'),
        yaxis2=dict(title='Revenue (₹)', side='right', overlaying='y'),
        hovermode='x unified'
    )
    
    fig.write_html('static/timeline_trends.html')
    
    # 2. Customer Acquisition
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_stats.index,
        y=monthly_stats['New_Customers'],
        name='New Customers',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Scatter(
        x=monthly_stats.index,
        y=monthly_stats['Unique_Customers'],
        name='Active Customers',
        line=dict(color='darkblue')
    ))
    
    fig.update_layout(
        title='Customer Acquisition and Retention',
        yaxis_title='Number of Customers',
        barmode='group',
        hovermode='x unified'
    )
    
    fig.write_html('static/customer_acquisition.html')
    
    # 3. Service Type Distribution Over Time
    service_cols = [col for col in monthly_stats.columns if col.startswith('Service_')]
    
    fig = go.Figure()
    for service in service_cols:
        fig.add_trace(go.Scatter(
            x=monthly_stats.index,
            y=monthly_stats[service],
            name=service.replace('Service_', ''),
            stackgroup='one',
            groupnorm='percent'
        ))
    
    fig.update_layout(
        title='Service Type Distribution Over Time',
        yaxis_title='Percentage of Orders',
        hovermode='x unified'
    )
    
    fig.write_html('static/service_distribution_timeline.html')
    
    # Create summary HTML
    create_timeline_summary(monthly_stats)

def create_timeline_summary(monthly_stats):
    """Create HTML summary of timeline statistics"""
    
    # Calculate overall statistics
    total_orders = monthly_stats['Order_Count'].sum()
    total_revenue = monthly_stats['Revenue'].sum()
    total_new_customers = monthly_stats['New_Customers'].sum()
    avg_monthly_orders = monthly_stats['Order_Count'].mean()
    avg_monthly_revenue = monthly_stats['Revenue'].mean()
    
    # Calculate growth rates
    first_month = monthly_stats.index.min().strftime('%B %Y')
    last_month = monthly_stats.index.max().strftime('%B %Y')
    total_months = len(monthly_stats)
    
    summary_html = f"""
    <div class="timeline-summary">
        <h3>Business Growth Summary ({first_month} to {last_month})</h3>
        <table class="table table-striped">
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Monthly Average</th>
            </tr>
            <tr>
                <td>Total Orders</td>
                <td>{total_orders:,.0f}</td>
                <td>{avg_monthly_orders:,.1f}</td>
            </tr>
            <tr>
                <td>Total Revenue</td>
                <td>₹{total_revenue:,.2f}</td>
                <td>₹{avg_monthly_revenue:,.2f}</td>
            </tr>
            <tr>
                <td>New Customers Acquired</td>
                <td>{total_new_customers:,.0f}</td>
                <td>{total_new_customers/total_months:,.1f}</td>
            </tr>
            <tr>
                <td>Average Order Value</td>
                <td>₹{(total_revenue/total_orders):,.2f}</td>
                <td>-</td>
            </tr>
        </table>
        
        <h4 class="mt-4">Recent Growth Rates (Last Month)</h4>
        <table class="table table-striped">
            <tr>
                <th>Metric</th>
                <th>Monthly Growth</th>
            </tr>
            <tr>
                <td>Revenue Growth</td>
                <td>{monthly_stats['Revenue_Growth'].iloc[-1]:+.1f}%</td>
            </tr>
            <tr>
                <td>Order Growth</td>
                <td>{monthly_stats['Order_Growth'].iloc[-1]:+.1f}%</td>
            </tr>
            <tr>
                <td>Customer Growth</td>
                <td>{monthly_stats['Customer_Growth'].iloc[-1]:+.1f}%</td>
            </tr>
        </table>
    </div>
    """
    
    with open('static/timeline_summary.html', 'w') as f:
        f.write(summary_html)

def main():
    print("Starting timeline analysis...")
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    try:
        # Load data
        data = pd.read_csv('311224.csv', encoding='windows-1252')
        print(f"Loaded {len(data)} records")
        
        # Print first few rows to debug column names
        print("\nFirst few rows:")
        print(data.head())
        print("\nColumns:", data.columns.tolist())
        
        # Create timeline statistics
        monthly_stats = create_timeline_statistics(data)
        
        # Save monthly statistics
        monthly_stats.to_csv('static/monthly_statistics.csv')
        
        print("Timeline analysis complete!")
        
    except Exception as e:
        print(f"Error in timeline analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 