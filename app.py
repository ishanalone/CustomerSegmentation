from flask import Flask, request, jsonify, send_file, current_app
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
from shutil import copyfile

def setup_static_files():
    """Setup static directory and copy necessary files"""
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
        print("Created static directory")
    
    # Copy index.html to static if it doesn't exist in static
    if not os.path.exists('static/index.html') and os.path.exists('index.html'):
        copyfile('index.html', 'static/index.html')
        print("Copied index.html to static directory")

# Create Flask app and setup static files
app = Flask(__name__)
setup_static_files()  # Now the function is defined before being called

# Load the trained model
try:
    model = joblib.load('customer_segmentation_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    print("Error: Model files not found. Please ensure model is trained and saved.")

@app.route('/segment', methods=['POST'])
def segment_customers():
    try:
        # Get data from POST request
        data = request.get_json()
        
        # Convert input data to DataFrame
        input_data = pd.DataFrame(data['customers'])
        
        # Calculate required metrics
        features = {
            'Order_Frequency': input_data['orders_count'],
            'Total_Spending': input_data['total_spending'],
            'Avg_Order_Value': input_data['total_spending'] / input_data['orders_count'],
            'Recency': (datetime.now() - pd.to_datetime(input_data['last_order_date'])).dt.days,
            'Avg_Pieces_Per_Order': input_data['total_pieces'] / input_data['orders_count'],
            'Avg_Weight_Per_Order': input_data['total_weight'] / input_data['orders_count'],
            'Avg_Discount_Rate': input_data['total_discount'] / input_data['total_spending'],
            'Delivery_Rate': input_data['delivered_orders'] / input_data['orders_count'],
            'Package_Usage_Rate': input_data['package_orders'] / input_data['orders_count'],
            'Avg_Monthly_Orders': input_data['orders_count'] / 12,
            'Value_Per_Kg': input_data['total_spending'] / input_data['total_weight']
        }
        
        # Create DataFrame with features
        feature_df = pd.DataFrame(features)
        
        # Handle missing values and infinities
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(0)
        
        # Scale the features
        scaled_data = scaler.transform(feature_df)
        
        # Perform segmentation
        segments = model.predict(scaled_data)
        
        # Marketing recommendations based on segments
        marketing_recommendations = {
            0: {  # High Value Customers
                "segment_name": "Premium Customers",
                "characteristics": "High spending, frequent orders, good package adoption",
                "recommendations": [
                    "Premium service packages with priority processing",
                    "Early access to new services and seasonal offers",
                    "Exclusive membership benefits and rewards",
                    "Personalized pick-up and delivery slots",
                    "Premium garment care recommendations"
                ],
                "marketing_channels": [
                    "Direct SMS/WhatsApp",
                    "Personal calls",
                    "Email newsletters with premium content",
                    "Exclusive app notifications"
                ],
                "retention_strategies": [
                    "VIP customer service",
                    "Birthday/anniversary special offers",
                    "Loyalty points multiplier",
                    "Feedback prioritization"
                ]
            },
            1: {  # Medium Value Customers
                "segment_name": "Growth Potential Customers",
                "characteristics": "Moderate spending, regular usage, price sensitive",
                "recommendations": [
                    "Value-added service bundles",
                    "Mid-tier package offerings",
                    "Seasonal discounts on bulk orders",
                    "Cross-sell additional services",
                    "Flexible service options"
                ],
                "marketing_channels": [
                    "SMS campaigns",
                    "App notifications",
                    "Email offers",
                    "Social media engagement"
                ],
                "retention_strategies": [
                    "Regular feedback collection",
                    "Service upgrade offers",
                    "Loyalty program benefits",
                    "Referral rewards"
                ]
            },
            2: {  # Regular Customers
                "segment_name": "Value Seekers",
                "characteristics": "Lower spending, occasional usage, high price sensitivity",
                "recommendations": [
                    "Introductory package offers",
                    "First-time service discounts",
                    "Basic service bundles",
                    "Special occasion promotions",
                    "Budget-friendly options"
                ],
                "marketing_channels": [
                    "Mass SMS campaigns",
                    "Social media ads",
                    "Local area promotions",
                    "Seasonal offers"
                ],
                "retention_strategies": [
                    "Service education",
                    "Price-sensitive packages",
                    "Easy service adoption",
                    "Flexible payment options"
                ]
            }
        }
        
        # Prepare detailed response
        customer_segments = []
        for i, segment in enumerate(segments):
            customer_info = {
                "customer_code": data['customers'][i].get('Customer_Code', f"Customer_{i}"),
                "segment": int(segment),
                "segment_info": marketing_recommendations[int(segment)],
                "metrics": {
                    "order_frequency": float(feature_df.iloc[i]['Order_Frequency']),
                    "total_spending": float(feature_df.iloc[i]['Total_Spending']),
                    "avg_order_value": float(feature_df.iloc[i]['Avg_Order_Value']),
                    "recency_days": float(feature_df.iloc[i]['Recency']),
                    "package_usage_rate": float(feature_df.iloc[i]['Package_Usage_Rate'])
                }
            }
            customer_segments.append(customer_info)
        
        response = {
            'status': 'success',
            'segmentation_results': customer_segments
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/customer_segments.csv')
def send_segments():
    try:
        # Get the absolute path to the file
        file_path = os.path.join(current_app.root_path, 'static', 'customer_segments.csv')
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='text/csv')
        else:
            current_app.logger.error(f"File not found: {file_path}")
            return "File not found", 404
    except Exception as e:
        current_app.logger.error(f"Error serving customer_segments.csv: {str(e)}")
        return str(e), 500

@app.route('/cluster_characteristics.csv')
def send_characteristics():
    try:
        file_path = os.path.join(current_app.root_path, 'static', 'cluster_characteristics.csv')
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='text/csv')
        else:
            current_app.logger.error(f"File not found: {file_path}")
            return "File not found", 404
    except Exception as e:
        current_app.logger.error(f"Error serving cluster_characteristics.csv: {str(e)}")
        return str(e), 500

@app.route('/')
def dashboard():
    try:
        # First try to serve from static directory
        return send_file('static/index.html')
    except:
        # If not found in static, try to serve from root directory
        return send_file('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    try:
        return send_file(f'static/{path}')
    except Exception as e:
        app.logger.error(f"Error serving {path}: {str(e)}")
        return str(e), 404

# Add this route for debugging
@app.route('/debug/files')
def debug_files():
    try:
        files = os.listdir('static')
        file_info = {}
        for file in files:
            path = os.path.join('static', file)
            file_info[file] = {
                'exists': os.path.exists(path),
                'size': os.path.getsize(path) if os.path.exists(path) else 0,
                'path': path
            }
        return jsonify({
            'files': file_info,
            'app_root': current_app.root_path,
            'static_folder': os.path.join(current_app.root_path, 'static')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add this route
@app.route('/churn-analysis')
def get_churn_analysis():
    try:
        churn_data = pd.read_csv('static/churn_analysis.csv')
        return jsonify({
            'status': 'success',
            'data': churn_data.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000) 