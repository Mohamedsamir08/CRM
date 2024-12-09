import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_absolute_error, classification_report
import base64

# Function to generate mock customer data based on the specified features
def generate_data():
    segments = ['Health-Conscious', 'Price-Sensitive', 'Premium']
    cities = ['Cairo', 'Alexandria', 'Giza', 'Sharm El Sheikh', 'Luxor', 'Aswan', 'Port Said', 'Tanta', 'Mansoura']
    profitability = ['Platinum', 'Gold', 'Iron']
    brands = ['ElMaleka', 'Meloky', 'Italiano']
    
    # Generate mock customer data
    customer_data = pd.DataFrame({
        'Customer ID': range(1, 501),
        'Segment': np.random.choice(segments, 500),
        'Purchase Frequency (Months)': np.random.randint(1, 12, 500),
        'Average Order Value (EGP)': np.random.randint(100, 1000, 500),
        'CLV (EGP)': np.random.randint(500, 5000, 500),
        'Churn Probability (%)': np.random.uniform(0, 100, 500),
        'Age': np.random.randint(18, 70, 500),
        'City': np.random.choice(cities, 500),
        'Profitability': np.random.choice(profitability, 500),
        'Product Preference': np.random.choice(brands, 500),
        'Order Volume': np.random.randint(1, 20, 500),  # Mock order volume
        'Redemption Rate': np.random.uniform(0, 1, 500),  # Mock redemption rate for offers
        'Feedback Score': np.random.randint(1, 6, 500),  # Mock customer feedback score (1-5)
        'Purchase Season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], 500)  # Mock seasonal trend
    })

    # Add Lifetime Value (LTV) based on average order value and purchase frequency (simplified)
    customer_data['Lifetime Value'] = customer_data['Average Order Value (EGP)'] * customer_data['Purchase Frequency (Months)'] * 12  # Annualized

    return customer_data

# Initialize customer data
customer_data = generate_data()

# Streamlit page configuration
st.set_page_config(page_title="Savola Egypt CRM Dashboard", layout="wide")

# Sidebar for navigation
st.sidebar.title("📊 CRM Dashboard")
page = st.sidebar.radio("🔍 Choose a Report", [
    "Customer Segmentation Report", 
    "Revenue Analysis Report", 
    "CLV Report", 
    "Churn Prediction Report", 
    "Marketing ROI Report", 
    "Purchase Frequency Analysis", 
    "Loyalty Program Participation Report",
])

# Function to add a new customer to the data
def add_new_customer(data):
    with st.sidebar.form(key='customer_form'):
        name = st.sidebar.text_input("Customer Name")
        email = st.sidebar.text_input("Email Address")
        age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1, value=30)
        city = st.sidebar.selectbox("City", options=customer_data['City'].unique())
        segment = st.sidebar.selectbox("Segment", options=customer_data['Segment'].unique())
        purchase_freq = st.sidebar.number_input("Purchase Frequency (Months)", min_value=1, max_value=12, step=1, value=6)
        avg_order = st.sidebar.number_input("Average Order Value (EGP)", min_value=100, max_value=1000, step=10, value=500)
        clv = st.sidebar.number_input("CLV (EGP)", min_value=500, max_value=5000, step=100, value=2000)
        churn_prob = st.sidebar.slider("Churn Probability (%)", 0.0, 100.0, 50.0, step=0.1)
        profitability = st.sidebar.selectbox("Profitability", options=customer_data['Profitability'].unique())
        product_pref = st.sidebar.selectbox("Product Preference", options=customer_data['Product Preference'].unique())
        order_volume = st.sidebar.number_input("Order Volume", min_value=1, max_value=20, step=1, value=5)
        redemption_rate = st.sidebar.slider("Redemption Rate", 0.0, 1.0, 0.5, step=0.01)
        feedback_score = st.sidebar.selectbox("Feedback Score", options=[1,2,3,4,5])
        purchase_season = st.sidebar.selectbox("Purchase Season", options=['Winter', 'Spring', 'Summer', 'Fall'])
        
        submit_button = st.sidebar.form_submit_button(label='Add Customer')
        
        if submit_button:
            new_id = data['Customer ID'].max() + 1
            new_customer = {
                'Customer ID': new_id,
                'Segment': segment,
                'Purchase Frequency (Months)': purchase_freq,
                'Average Order Value (EGP)': avg_order,
                'CLV (EGP)': clv,
                'Churn Probability (%)': churn_prob,
                'Age': age,
                'City': city,
                'Profitability': profitability,
                'Product Preference': product_pref,
                'Order Volume': order_volume,
                'Redemption Rate': redemption_rate,
                'Feedback Score': feedback_score,
                'Purchase Season': purchase_season,
                'Lifetime Value': avg_order * purchase_freq * 12
            }
            data = data.append(new_customer, ignore_index=True)
            st.sidebar.success(f"Customer {name} added successfully!")
    return data

# 1. Customer Segmentation Report
if page == "Customer Segmentation Report":
    st.markdown("<h1 style='text-align: center;'>👥 Customer Segmentation Report</h1>", unsafe_allow_html=True)
    
    # Filters
    st.sidebar.header("🔧 Filters")
    selected_city = st.sidebar.multiselect("Select City", options=customer_data['City'].unique(), default=customer_data['City'].unique())
    selected_segment = st.sidebar.multiselect("Select Segment", options=customer_data['Segment'].unique(), default=customer_data['Segment'].unique())
    selected_profitability = st.sidebar.multiselect("Select Profitability", options=customer_data['Profitability'].unique(), default=customer_data['Profitability'].unique())
    
    # Filter data based on selections
    filtered_data = customer_data[
        (customer_data['City'].isin(selected_city)) &
        (customer_data['Segment'].isin(selected_segment)) &
        (customer_data['Profitability'].isin(selected_profitability))
    ]
    
    st.markdown("### 1. Demographic Segmentation Analysis")
    demographic_dist = px.histogram(filtered_data, x='Age', nbins=20, title='Age Distribution', color='Segment')
    st.plotly_chart(demographic_dist, use_container_width=True)
    
    st.markdown("### 2. Behavioral Segmentation Analysis")
    rfm = filtered_data.groupby('Customer ID').agg({
        'Purchase Frequency (Months)': 'mean',
        'Average Order Value (EGP)': 'mean',
        'CLV (EGP)': 'sum'
    }).reset_index()
    rfm['R'] = pd.qcut(rfm['Purchase Frequency (Months)'], 4, labels=False)
    rfm['F'] = pd.qcut(rfm['Average Order Value (EGP)'], 4, labels=False)
    rfm['M'] = pd.qcut(rfm['CLV (EGP)'], 4, labels=False)
    st.write(rfm.head())
    st.markdown("#### RFM Scatter Plot")
    rfm_fig = px.scatter(rfm, x='F', y='M', color='R', title='RFM Segmentation')
    st.plotly_chart(rfm_fig, use_container_width=True)
    
    st.markdown("### 3. Geographic Segmentation Analysis")
    geo_dist = filtered_data['City'].value_counts().reset_index()
    geo_dist.columns = ['City', 'Customer Count']
    geo_map = px.choropleth(filtered_data, locations='City',
                            locationmode='geojson-id',
                            color='Customer ID',
                            title='Geographic Distribution of Customers')
    st.plotly_chart(geo_map, use_container_width=True)
    
    st.markdown("### 4. Psychographic Segmentation Analysis")
    psychographic_dist = px.bar(filtered_data, x='Segment', y='Customer ID', color='Segment',
                                title='Psychographic Segmentation')
    st.plotly_chart(psychographic_dist, use_container_width=True)
    
    st.markdown("### 5. Profitability Segmentation Analysis")
    profitability_seg = filtered_data.groupby('Profitability')['CLV (EGP)'].mean().reset_index()
    profitability_fig = px.bar(profitability_seg, x='Profitability', y='CLV (EGP)', 
                                title='Profitability Segmentation', color='Profitability')
    st.plotly_chart(profitability_fig, use_container_width=True)
    
    st.markdown("### 📈 Customer Segmentation Overview")
    st.dataframe(filtered_data[['Customer ID', 'Segment', 'Profitability', 'City', 'CLV (EGP)']].head(10))

# 2. Revenue Analysis Report
if page == "Revenue Analysis Report":
    st.markdown("<h1 style='text-align: center;'>💰 Revenue Analysis Report</h1>", unsafe_allow_html=True)
    
    # 1. Revenue Trend Analysis
    st.markdown("### 1. Revenue Trend Analysis")
    revenue_trend = customer_data.groupby('Purchase Season')['CLV (EGP)'].sum().reset_index()
    revenue_trend_fig = px.line(revenue_trend, x='Purchase Season', y='CLV (EGP)', title='Revenue Trend by Season')
    st.plotly_chart(revenue_trend_fig, use_container_width=True)
    
    # 2. Product-Line Revenue Contribution
    st.markdown("### 2. Product-Line Revenue Contribution")
    product_revenue = customer_data.groupby('Product Preference')['CLV (EGP)'].sum().reset_index()
    product_revenue_fig = px.pie(product_revenue, names='Product Preference', values='CLV (EGP)', title='Revenue Contribution by Product Line')
    st.plotly_chart(product_revenue_fig, use_container_width=True)
    
    # 3. Revenue by Customer Segment
    st.markdown("### 3. Revenue by Customer Segment")
    revenue_segment = customer_data.groupby('Segment')['CLV (EGP)'].sum().reset_index()
    revenue_segment_fig = px.bar(revenue_segment, x='Segment', y='CLV (EGP)', color='Segment', title='Revenue by Customer Segment')
    st.plotly_chart(revenue_segment_fig, use_container_width=True)
    
    # 4. Revenue per Channel Analysis
    st.markdown("### 4. Revenue per Channel Analysis")
    # Assuming channels based on 'City' or 'Segment'
    channel_revenue = customer_data.groupby('Segment')['CLV (EGP)'].sum().reset_index()
    channel_fig = px.bar(channel_revenue, x='Segment', y='CLV (EGP)', color='Segment', title='Revenue by Channel (Segment)')
    st.plotly_chart(channel_fig, use_container_width=True)
    
    # 5. Seasonal Revenue Analysis
    st.markdown("### 5. Seasonal Revenue Analysis")
    seasonal_revenue = customer_data.groupby('Purchase Season')['CLV (EGP)'].sum().reset_index()
    seasonal_fig = px.bar(seasonal_revenue, x='Purchase Season', y='CLV (EGP)', color='Purchase Season', title='Seasonal Revenue Analysis')
    st.plotly_chart(seasonal_fig, use_container_width=True)
    
    st.markdown("### 📊 Detailed Revenue Data")
    st.dataframe(customer_data[['Customer ID', 'Segment', 'Product Preference', 'Purchase Season', 'CLV (EGP)']].head(10))

# 3. CLV Report
if page == "CLV Report":
    st.markdown("<h1 style='text-align: center;'>💰 CLV Report</h1>", unsafe_allow_html=True)
    
    # 1. CLV Distribution Analysis
    st.markdown("### 1. CLV Distribution Analysis")
    clv_dist = px.histogram(customer_data, x='CLV (EGP)', nbins=30, title='CLV Distribution', color_discrete_sequence=['#FFA07A'])
    st.plotly_chart(clv_dist, use_container_width=True)
    
    # 2. CLV by Customer Segment
    st.markdown("### 2. CLV by Customer Segment")
    clv_segment = customer_data.groupby('Segment')['CLV (EGP)'].mean().reset_index()
    clv_segment_fig = px.bar(clv_segment, x='Segment', y='CLV (EGP)', color='Segment', title='Average CLV by Segment', text='CLV (EGP)')
    clv_segment_fig.update_traces(texttemplate='%{text:.2s}', textposition='inside')
    st.plotly_chart(clv_segment_fig, use_container_width=True)
    
    # 3. CLV Prediction Modeling
    st.markdown("### 3. CLV Prediction Modeling")
    features = ['Purchase Frequency (Months)', 'Average Order Value (EGP)', 'Age', 'Order Volume', 'Redemption Rate', 'Feedback Score']
    X = customer_data[features]
    y = customer_data['CLV (EGP)']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    
    # Model Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f} EGP")
    
    # Feature Importance
    importance = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
    st.markdown("#### 3.1 Feature Importance")
    importance_fig = px.bar(importance, x=importance.values, y=importance.index, orientation='h', title='Feature Importance in CLV Prediction')
    st.plotly_chart(importance_fig, use_container_width=True)
    
    # 4. Factors Influencing CLV
    st.markdown("### 4. Factors Influencing CLV")
    corr_matrix = customer_data[features + ['CLV (EGP)']].corr()
    st.write("#### Correlation Matrix")
    st.write(corr_matrix)
    st.write("#### Heatmap")
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt.gcf())
    
    # 5. CLV vs. Acquisition Cost Analysis
    st.markdown("### 5. CLV vs. Acquisition Cost Analysis")
    # Assuming Acquisition Cost as a random variable for demonstration
    customer_data['Acquisition Cost (EGP)'] = np.random.randint(100, 1000, 500)
    clv_cac = customer_data[['CLV (EGP)', 'Acquisition Cost (EGP)']].copy()
    clv_cac_fig = px.scatter(clv_cac, x='Acquisition Cost (EGP)', y='CLV (EGP)', trendline='ols',
                             title='CLV vs. Acquisition Cost')
    st.plotly_chart(clv_cac_fig, use_container_width=True)
    
    st.markdown("### 📈 CLV Overview")
    st.dataframe(customer_data[['Customer ID', 'CLV (EGP)', 'Purchase Frequency (Months)', 'Average Order Value (EGP)']].head(10))
    
# 4. Churn Prediction Report
if page == "Churn Prediction Report":
    st.markdown("<h1 style='text-align: center;'>🚪 Churn Prediction Report</h1>", unsafe_allow_html=True)
    
    # 1. Churn Rate Analysis
    st.markdown("### 1. Churn Rate Analysis")
    churn_rate = customer_data['Churn Probability (%)'].mean()
    st.write(f"**Overall Churn Rate:** {churn_rate:.2f}%")
    churn_dist = px.histogram(customer_data, x='Churn Probability (%)', nbins=30, title='Churn Probability Distribution', color_discrete_sequence=['#FF6347'])
    st.plotly_chart(churn_dist, use_container_width=True)
    
    # 2. Churn Risk Segmentation
    st.markdown("### 2. Churn Risk Segmentation")
    customer_data['Churn Risk'] = pd.cut(customer_data['Churn Probability (%)'], bins=[0, 33, 66, 100], labels=['Low', 'Medium', 'High'])
    churn_risk = customer_data['Churn Risk'].value_counts().reset_index()
    churn_risk.columns = ['Churn Risk', 'Count']
    churn_risk_fig = px.pie(churn_risk, names='Churn Risk', values='Count', title='Churn Risk Segmentation')
    st.plotly_chart(churn_risk_fig, use_container_width=True)
    
    # 3. Predictive Modeling for Churn
    st.markdown("### 3. Predictive Modeling for Churn")
    # Define target variable
    customer_data['Churn'] = customer_data['Churn Probability (%)'].apply(lambda x: 1 if x > 50 else 0)
    X = customer_data[['Purchase Frequency (Months)', 'Average Order Value (EGP)', 'Age', 'Order Volume', 'Redemption Rate', 'Feedback Score']]
    y = customer_data['Churn']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_clf = clf_model.predict(X_test)
    
    # Model Evaluation
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred_clf))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_clf)
    cm_fig = ff.create_annotated_heatmap(cm, x=['Predicted No', 'Predicted Yes'], y=['Actual No', 'Actual Yes'],
                                        colorscale='Viridis', showscale=True)
    cm_fig.update_layout(title='Confusion Matrix')
    st.plotly_chart(cm_fig, use_container_width=True)
    
    # 5. Impact of Marketing Interventions on Churn
    st.markdown("### 5. Impact of Marketing Interventions on Churn")
    # Simulate before and after marketing intervention
    before = customer_data[customer_data['Churn'] == 1].shape[0]
    after = int(before * 0.8)  # Assume 20% reduction
    st.write(f"**Churn Before Intervention:** {before}")
    st.write(f"**Churn After Intervention:** {after}")
    impact_fig = px.bar(x=['Before', 'After'], y=[before, after], labels={'x': 'Intervention', 'y': 'Churn Count'},
                        title='Impact of Marketing Interventions on Churn')
    st.plotly_chart(impact_fig, use_container_width=True)
    
    st.markdown("### 📉 Churn Overview")
    st.dataframe(customer_data[['Customer ID', 'Churn Probability (%)', 'Churn Risk']].head(10))

# 5. Marketing ROI Report
if page == "Marketing ROI Report":
    st.markdown("<h1 style='text-align: center;'>📈 Marketing ROI Report</h1>", unsafe_allow_html=True)
    
    # 1. Campaign Performance Analysis
    st.markdown("### 1. Campaign Performance Analysis")
    # Simulate campaign data
    campaigns = ['Ramadan', 'Summer Sale', 'Back to School', 'New Year', 'Easter']
    campaign_data = pd.DataFrame({
        'Campaign': np.random.choice(campaigns, 100),
        'Reach': np.random.randint(1000, 10000, 100),
        'Engagement': np.random.randint(100, 5000, 100),
        'Conversions': np.random.randint(50, 1000, 100),
        'Cost (EGP)': np.random.randint(5000, 50000, 100)
    })
    campaign_summary = campaign_data.groupby('Campaign').agg({
        'Reach': 'sum',
        'Engagement': 'sum',
        'Conversions': 'sum',
        'Cost (EGP)': 'sum'
    }).reset_index()
    campaign_summary['ROI'] = (campaign_summary['Conversions'] * 100) / campaign_summary['Cost (EGP)']
    campaign_fig = px.bar(campaign_summary, x='Campaign', y='ROI', color='Campaign',
                          title='ROI by Marketing Campaign', text='ROI')
    st.plotly_chart(campaign_fig, use_container_width=True)
    
    # 2. ROI by Marketing Channel
    st.markdown("### 2. ROI by Marketing Channel")
    channels = ['Social Media', 'Email', 'Offline', 'Influencer', 'SEO']
    channel_data = pd.DataFrame({
        'Channel': np.random.choice(channels, 100),
        'Conversions': np.random.randint(50, 1000, 100),
        'Cost (EGP)': np.random.randint(5000, 50000, 100)
    })
    channel_summary = channel_data.groupby('Channel').agg({
        'Conversions': 'sum',
        'Cost (EGP)': 'sum'
    }).reset_index()
    channel_summary['ROI'] = (channel_summary['Conversions'] * 100) / channel_summary['Cost (EGP)']
    channel_fig = px.pie(channel_summary, names='Channel', values='ROI', title='ROI by Marketing Channel')
    st.plotly_chart(channel_fig, use_container_width=True)
    
    # 3. Cost-Benefit Analysis of Loyalty Programs
    st.markdown("### 3. Cost-Benefit Analysis of Loyalty Programs")
    loyalty_cost = 500000  # Example value
    loyalty_benefit = customer_data['CLV (EGP)'].sum() * 0.2  # Assuming 20% increase
    st.write(f"**Total Cost of Loyalty Program:** EGP {loyalty_cost:,}")
    st.write(f"**Total Benefit from Loyalty Program:** EGP {loyalty_benefit:,.2f}")
    roi = (loyalty_benefit - loyalty_cost) / loyalty_cost * 100
    st.write(f"**ROI:** {roi:.2f}%")
    cost_benefit_fig = px.bar(x=['Cost', 'Benefit'], y=[loyalty_cost, loyalty_benefit],
                              labels={'x': 'Category', 'y': 'Amount (EGP)'},
                              title='Cost-Benefit Analysis of Loyalty Programs')
    st.plotly_chart(cost_benefit_fig, use_container_width=True)
    
    # 4. Attribution Modeling
    st.markdown("### 4. Attribution Modeling")
    # Simulate attribution data
    attribution_data = pd.DataFrame({
        'Touchpoint': np.random.choice(['Social Media', 'Email', 'Offline', 'Influencer', 'SEO'], 500),
        'Conversions': np.random.randint(1, 10, 500)
    })
    attribution_summary = attribution_data.groupby('Touchpoint')['Conversions'].sum().reset_index()
    attribution_fig = px.bar(attribution_summary, x='Touchpoint', y='Conversions', color='Touchpoint',
                              title='Attribution Modeling - Conversions by Touchpoint')
    st.plotly_chart(attribution_fig, use_container_width=True)
    
    st.markdown("### 📈 Marketing ROI Overview")
    st.dataframe(campaign_summary)

# 6. Purchase Frequency Analysis
if page == "Purchase Frequency Analysis":
    st.markdown("<h1 style='text-align: center;'>🔄 Purchase Frequency Analysis</h1>", unsafe_allow_html=True)
    
    # 1. Frequency Distribution Analysis
    st.markdown("### 1. Frequency Distribution Analysis")
    freq_dist = px.histogram(customer_data, x='Purchase Frequency (Months)', nbins=12, title='Purchase Frequency Distribution', color='Segment')
    st.plotly_chart(freq_dist, use_container_width=True)
    
    # 2. Frequency Trends Over Time
    st.markdown("### 2. Frequency Trends Over Time")
    # Assuming 'Purchase Season' as time indicator
    freq_trend = customer_data.groupby('Purchase Season')['Purchase Frequency (Months)'].mean().reset_index()
    freq_trend_fig = px.line(freq_trend, x='Purchase Season', y='Purchase Frequency (Months)', title='Average Purchase Frequency Over Seasons')
    st.plotly_chart(freq_trend_fig, use_container_width=True)
    
    # 3. Frequency by Customer Segment
    st.markdown("### 3. Frequency by Customer Segment")
    freq_segment = customer_data.groupby('Segment')['Purchase Frequency (Months)'].mean().reset_index()
    freq_segment_fig = px.bar(freq_segment, x='Segment', y='Purchase Frequency (Months)', color='Segment',
                               title='Average Purchase Frequency by Segment')
    st.plotly_chart(freq_segment_fig, use_container_width=True)
    
    
    # 5. Impact of Promotions on Purchase Frequency
    st.markdown("### 5. Impact of Promotions on Purchase Frequency")
    # Simulate promotion impact
    promotion_data = pd.DataFrame({
        'Promotion': np.random.choice(['Yes', 'No'], 500, p=[0.3, 0.7]),
        'Purchase Frequency (Months)': np.random.randint(1, 12, 500)
    })
    promotion_group = promotion_data.groupby('Promotion')['Purchase Frequency (Months)'].mean().reset_index()
    promotion_fig = px.bar(promotion_group, x='Promotion', y='Purchase Frequency (Months)', color='Promotion',
                           title='Impact of Promotions on Purchase Frequency')
    st.plotly_chart(promotion_fig, use_container_width=True)
    
    st.markdown("### 🔄 Purchase Frequency Overview")
    st.dataframe(customer_data[['Customer ID', 'Purchase Frequency (Months)', 'Segment']].head(10))

# 7. Loyalty Program Participation Report
if page == "Loyalty Program Participation Report":
    st.markdown("<h1 style='text-align: center;'>🎁 Loyalty Program Participation Report</h1>", unsafe_allow_html=True)
    
    # 1. Enrollment Rate Analysis
    st.markdown("### 1. Enrollment Rate Analysis")
    # Simulate enrollment data
    customer_data['Enrolled'] = np.random.choice(['Yes', 'No'], 500, p=[0.6, 0.4])
    enrollment_rate = customer_data['Enrolled'].value_counts().reset_index()
    enrollment_rate.columns = ['Enrolled', 'Count']
    enrollment_fig = px.pie(enrollment_rate, names='Enrolled', values='Count', title='Enrollment Rate in Loyalty Program')
    st.plotly_chart(enrollment_fig, use_container_width=True)
    
    # 2. Tier Distribution Analysis
    st.markdown("### 2. Tier Distribution Analysis")
    # Simulate tiers
    tiers = ['Bronze', 'Silver', 'Gold', 'Premium']
    customer_data['Tier'] = np.where(customer_data['Enrolled'] == 'Yes', 
                                     np.random.choice(tiers, 500, p=[0.5, 0.3, 0.15, 0.05]), 
                                     'None')
    tier_distribution = customer_data['Tier'].value_counts().reset_index()
    tier_distribution.columns = ['Tier', 'Count']
    tier_fig = px.pie(tier_distribution, names='Tier', values='Count', title='Tier Distribution in Loyalty Program')
    st.plotly_chart(tier_fig, use_container_width=True)
    
    # 3. Reward Redemption Patterns
    st.markdown("### 3. Reward Redemption Patterns")
    redemption = customer_data[customer_data['Enrolled'] == 'Yes']
    redemption_fig = px.histogram(redemption, x='Redemption Rate', nbins=30, title='Reward Redemption Rate Distribution', color='Tier')
    st.plotly_chart(redemption_fig, use_container_width=True)
    
    # 4. Impact of Loyalty Program on CLV
    st.markdown("### 4. Impact of Loyalty Program on CLV")
    loyalty_clv = customer_data.groupby('Enrolled')['CLV (EGP)'].mean().reset_index()
    loyalty_clv_fig = px.bar(loyalty_clv, x='Enrolled', y='CLV (EGP)', color='Enrolled',
                             title='Average CLV: Enrolled vs. Non-Enrolled')
    st.plotly_chart(loyalty_clv_fig, use_container_width=True)
    
    # 5. Customer Satisfaction and Loyalty Engagement
    st.markdown("### 5. Customer Satisfaction and Loyalty Engagement")
    satisfaction = customer_data[customer_data['Enrolled'] == 'Yes']
    satisfaction_fig = px.box(satisfaction, x='Tier', y='Feedback Score', title='Feedback Score by Loyalty Tier')
    st.plotly_chart(satisfaction_fig, use_container_width=True)
    
    st.markdown("### 🎁 Loyalty Program Overview")
    st.dataframe(customer_data[['Customer ID', 'Enrolled', 'Tier', 'Feedback Score']].head(10))

# 8. Download Data Option
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Download Data")
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(customer_data)
st.sidebar.download_button(
    label="Download Customer Data as CSV",
    data=csv,
    file_name='customer_data.csv',
    mime='text/csv',
)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>© 2024 Savola Egypt CRM Dashboard. All rights reserved.</p>", unsafe_allow_html=True)
