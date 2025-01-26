import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

def load_and_prepare_data():
    # Load the segmented customer data
    customer_segments = pd.read_csv('static/customer_segments.csv')
    cluster_characteristics = pd.read_csv('static/cluster_characteristics.csv')
    
    # Convert segment to string for better visualization
    customer_segments['Segment'] = customer_segments['Segment'].map({
        0: 'High Value',
        1: 'Medium Value',
        2: 'Regular'
    })
    
    return customer_segments, cluster_characteristics

def create_radar_chart(customer_segments, prefix=''):
    """Create radar chart showing segment characteristics"""
    # Determine segment column name
    segment_col = 'RF_Segment' if 'RF_Segment' in customer_segments.columns else 'Segment'
    
    # Map numeric segments to labels if needed
    if customer_segments[segment_col].dtype in ['int64', 'int32']:
        customer_segments[segment_col] = customer_segments[segment_col].map({
            0: 'High Value',
            1: 'Medium Value',
            2: 'Regular'
        })
    
    # Calculate mean values for each segment
    segment_means = customer_segments.groupby(segment_col)[
        ['Order_Frequency', 'Total_Spending', 'Avg_Order_Value',
         'Laundry_Ratio', 'DryClean_Ratio', 'Package_Usage_Rate']
    ].mean()
    
    # Scale the values for better visualization
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(segment_means)
    scaled_df = pd.DataFrame(scaled_values, 
                           index=segment_means.index, 
                           columns=segment_means.columns)
    
    # Create radar chart
    categories = ['Order Frequency', 'Total Spending', 'Avg Order Value',
                 'Laundry Usage', 'Dry Clean Usage', 'Package Usage']
    
    fig = go.Figure()
    
    colors = {'High Value': '#FF6B6B', 'Medium Value': '#4ECDC4', 'Regular': '#45B7D1'}
    
    for segment in scaled_df.index:
        values = scaled_df.loc[segment].values.tolist()
        # Add the first value again to close the polygon
        values.append(values[0])
        categories_plot = categories + [categories[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_plot,
            fill='toself',
            name=segment,
            line_color=colors.get(segment)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-2, 2]  # Standardized scale
            )
        ),
        showlegend=True,
        title='Customer Segment Characteristics',
        height=500,
        width=700,
        template='plotly_white'
    )
    
    # Save as standalone HTML with prefix
    output_file = f'static/{prefix}radar_chart.html'
    fig.write_html(output_file, 
                  full_html=True, 
                  include_plotlyjs=True,
                  config={'displayModeBar': False})

def create_spending_frequency_plot(customer_segments, prefix=''):
    # Determine segment column name
    segment_col = 'RF_Segment' if 'RF_Segment' in customer_segments.columns else 'Segment'
    
    # Map segments if needed
    if customer_segments[segment_col].dtype in ['int64', 'int32']:
        customer_segments[segment_col] = customer_segments[segment_col].map({
            0: 'High Value',
            1: 'Medium Value',
            2: 'Regular'
        })
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=customer_segments,
        x='Total_Spending',
        y='Order_Frequency',
        hue=segment_col,  # Use the determined segment column
        size='Avg_Order_Value',
        sizes=(50, 400),
        alpha=0.6
    )
    plt.title('Customer Segments: Spending vs Frequency')
    plt.xlabel('Total Spending')
    plt.ylabel('Order Frequency')
    plt.savefig(f'static/{prefix}spending_frequency.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_recency_value_plot(customer_segments):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=customer_segments,
        x='Recency',
        y='Value_Per_Kg',
        hue='Segment',
        size='Total_Spending',
        sizes=(50, 400),
        alpha=0.6
    )
    plt.title('Customer Segments: Recency vs Value per Kg')
    plt.xlabel('Recency (days)')
    plt.ylabel('Value per Kg')
    plt.savefig('static/recency_value.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_segment_distribution(customer_segments, prefix=''):
    """Create distribution plots for key metrics across segments"""
    # Determine segment column name
    segment_col = 'RF_Segment' if 'RF_Segment' in customer_segments.columns else 'Segment'
    
    # Map segments if needed
    if customer_segments[segment_col].dtype in ['int64', 'int32']:
        customer_segments[segment_col] = customer_segments[segment_col].map({
            0: 'High Value',
            1: 'Medium Value',
            2: 'Regular'
        })
    
    # Create a 2x2 subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Order Frequency by Segment',
            'Total Spending by Segment',
            'Average Order Value by Segment',
            'Package Usage Rate by Segment'
        )
    )
    
    metrics = {
        'Order_Frequency': {'row': 1, 'col': 1},
        'Total_Spending': {'row': 1, 'col': 2},
        'Avg_Order_Value': {'row': 2, 'col': 1},
        'Package_Usage_Rate': {'row': 2, 'col': 2}
    }
    
    colors = {'High Value': '#FF6B6B', 'Medium Value': '#4ECDC4', 'Regular': '#45B7D1'}
    
    for metric, pos in metrics.items():
        for segment in customer_segments[segment_col].unique():
            segment_data = customer_segments[customer_segments[segment_col] == segment][metric]
            
            fig.add_trace(
                go.Histogram(
                    x=segment_data,
                    name=f"{segment} - {metric}",
                    marker_color=colors.get(segment),
                    opacity=0.7,
                    nbinsx=30,
                    showlegend=(pos['row'] == 1 and pos['col'] == 1)
                ),
                row=pos['row'],
                col=pos['col']
            )
            
            fig.update_xaxes(title_text=metric, row=pos['row'], col=pos['col'])
            fig.update_yaxes(title_text='Count', row=pos['row'], col=pos['col'])

    fig.update_layout(
        height=800,
        width=1200,
        title_text="Segment Distributions",
        showlegend=True,
        template='plotly_white'
    )
    
    # Save with prefix
    fig.write_html(f'static/{prefix}segment_distributions.html')

def create_service_preference_chart(customer_segments, prefix=''):
    # Determine segment column name
    segment_col = 'RF_Segment' if 'RF_Segment' in customer_segments.columns else 'Segment'
    
    # Map segments if needed
    if customer_segments[segment_col].dtype in ['int64', 'int32']:
        customer_segments[segment_col] = customer_segments[segment_col].map({
            0: 'High Value',
            1: 'Medium Value',
            2: 'Regular'
        })
    
    service_dist = pd.crosstab(customer_segments[segment_col], 
                              customer_segments['Primary_Service'])
    
    plt.figure(figsize=(12, 6))
    service_dist.plot(kind='bar', stacked=True)
    plt.title('Service Preferences by Segment')
    plt.xlabel('Customer Segment')
    plt.ylabel('Number of Customers')
    plt.legend(title='Service Type', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(f'static/{prefix}service_preferences.png')
    plt.close()

def generate_segment_insights(customer_segments, prefix=''):
    """Generate detailed insights for each segment"""
    # Determine segment column name
    segment_col = 'RF_Segment' if 'RF_Segment' in customer_segments.columns else 'Segment'
    
    # Map segments if needed
    if customer_segments[segment_col].dtype in ['int64', 'int32']:
        customer_segments[segment_col] = customer_segments[segment_col].map({
            0: 'High Value',
            1: 'Medium Value',
            2: 'Regular'
        })
    
    # Calculate comprehensive metrics for each segment
    insights = customer_segments.groupby(segment_col).agg({
        'Order_Frequency': ['mean', 'median', 'count'],
        'Total_Spending': ['mean', 'median', 'sum'],
        'Avg_Order_Value': ['mean', 'median'],
        'Recency': 'mean',
        'Package_Usage_Rate': 'mean',
        'Laundry_Ratio': 'mean',
        'DryClean_Ratio': 'mean',
        'Value_Per_Kg': 'mean'
    }).round(2)
    
    # Flatten column names
    insights.columns = [
        'Avg Orders', 'Median Orders', 'Customer Count',
        'Avg Spending', 'Median Spending', 'Total Revenue',
        'Avg Order Value', 'Median Order Value',
        'Avg Recency (days)',
        'Package Usage Rate',
        'Laundry Service Ratio',
        'Dry Clean Service Ratio',
        'Value per Kg'
    ]
    
    # Add percentage of total customers
    total_customers = insights['Customer Count'].sum()
    insights['Customer %'] = (insights['Customer Count'] / total_customers * 100).round(1)
    
    # Format currency values
    currency_cols = ['Avg Spending', 'Median Spending', 'Total Revenue', 'Avg Order Value', 'Median Order Value']
    for col in currency_cols:
        insights[col] = insights[col].apply(lambda x: f'₹{x:,.2f}')
    
    # Format percentage values
    pct_cols = ['Package Usage Rate', 'Laundry Service Ratio', 'Dry Clean Service Ratio', 'Customer %']
    for col in pct_cols:
        insights[col] = insights[col].apply(lambda x: f'{x:.1f}%')
    
    # Create a styled HTML table
    html_table = """
    <style>
        .insights-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }
        .insights-table th, .insights-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .insights-table th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        .insights-table tr:hover {
            background-color: #f5f5f5;
        }
    </style>
    """
    html_table += insights.to_html(classes='insights-table')
    
    # Save to file with prefix
    with open(f'static/{prefix}segment_insights.html', 'w') as f:
        f.write(html_table)

def create_visualization_data(prefix=''):
    customer_segments, _ = load_and_prepare_data()
    
    # Create data for radar chart
    radar_data = create_radar_chart_data(customer_segments)
    with open(f'static/{prefix}radar_chart_data.json', 'w') as f:
        json.dump(radar_data, f)
    
    # Create data for spending frequency chart
    spending_freq_data = create_spending_frequency_data(customer_segments)
    with open(f'static/{prefix}spending_frequency_data.json', 'w') as f:
        json.dump(spending_freq_data, f)
    
    # Create data for service preferences chart
    service_pref_data = create_service_preference_data(customer_segments)
    with open(f'static/{prefix}service_preferences_data.json', 'w') as f:
        json.dump(service_pref_data, f)
    
    # Create data for segment distribution
    segment_dist_data = create_segment_distribution_data(customer_segments)
    with open(f'static/{prefix}segment_distribution_data.json', 'w') as f:
        json.dump(segment_dist_data, f)

def create_radar_chart_data(customer_segments):
    """Create data for the radar chart visualization"""
    # Calculate mean values for each segment
    segment_means = customer_segments.groupby('Segment')[
        ['Order_Frequency', 'Total_Spending', 'Avg_Order_Value',
         'Laundry_Ratio', 'DryClean_Ratio', 'Package_Usage_Rate']
    ].mean()
    
    # Convert to dictionary format for JSON
    radar_data = {
        'categories': ['Order Frequency', 'Total Spending', 'Avg Order Value',
                      'Laundry Usage', 'Dry Clean Usage', 'Package Usage'],
        'segments': {}
    }
    
    for segment in segment_means.index:
        values = segment_means.loc[segment].values.tolist()
        radar_data['segments'][segment] = values
    
    return radar_data

def create_spending_frequency_data(customer_segments):
    """Create data for the spending vs frequency scatter plot"""
    return {
        'data': customer_segments[['Segment', 'Total_Spending', 
                                 'Order_Frequency', 'Avg_Order_Value']]
        .to_dict(orient='records')
    }

def create_service_preference_data(customer_segments):
    """Create data for the service preferences chart"""
    service_dist = pd.crosstab(
        customer_segments['Segment'],
        customer_segments['Primary_Service']
    )
    
    return {
        'segments': service_dist.index.tolist(),
        'services': service_dist.columns.tolist(),
        'data': service_dist.values.tolist()
    }

def create_segment_distribution_data(customer_segments):
    """Create data for the segment distribution histograms"""
    metrics = ['Order_Frequency', 'Total_Spending', 
              'Avg_Order_Value', 'Package_Usage_Rate']
    
    distribution_data = {}
    for metric in metrics:
        distribution_data[metric] = {
            segment: customer_segments[
                customer_segments['Segment'] == segment
            ][metric].tolist()
            for segment in customer_segments['Segment'].unique()
        }
    
    return distribution_data

def create_rf_visualizations(rf_customer_metrics, feature_importance):
    """Create visualizations for Random Forest model"""
    # Create radar chart for RF
    create_radar_chart(rf_customer_metrics, prefix='rf_')
    
    # Create spending frequency plot for RF
    create_spending_frequency_plot(rf_customer_metrics, prefix='rf_')
    
    # Create segment distribution for RF
    create_segment_distribution(rf_customer_metrics, prefix='rf_')
    
    # Create service preference chart for RF
    create_service_preference_chart(rf_customer_metrics, prefix='rf_')
    
    # Generate insights for RF
    generate_segment_insights(rf_customer_metrics, prefix='rf_')
    
    # Create feature importance plot
    create_feature_importance_plot(feature_importance)

def create_feature_importance_plot(feature_importance):
    """Create feature importance visualization"""
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importance.head(10),
        x='importance',
        y='feature'
    )
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('static/rf_feature_importance.png')
    plt.close()

def create_visualizations(customer_segments, prefix=''):
    """Create all visualizations with optional prefix for filenames"""
    # Create radar chart
    create_radar_chart(customer_segments)
    
    # Create spending frequency plot
    create_spending_frequency_plot(customer_segments)
    
    # Create recency value plot
    create_recency_value_plot(customer_segments)
    
    # Create segment distribution
    create_segment_distribution(customer_segments)
    
    # Create service preference chart
    create_service_preference_chart(customer_segments)
    
    # Generate insights
    generate_segment_insights(customer_segments)
    
    # Create visualization data
    create_visualization_data()

def main():
    print("Starting visualization generation...")
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Generate KMeans visualizations
    customer_segments, cluster_characteristics = load_and_prepare_data()
    create_visualizations(customer_segments)
    
    # Generate Random Forest visualizations
    rf_customer_metrics = pd.read_csv('static/rf_customer_segments.csv')
    feature_importance = pd.read_csv('static/rf_feature_importance.csv')
    create_rf_visualizations(rf_customer_metrics, feature_importance)
    
    print("Visualization generation complete!")

if __name__ == "__main__":
    main() 