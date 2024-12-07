import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

# Function to generate mock customer data
def generate_data():
    segments = ['Price-Sensitive', 'Innovation-Oriented', 'Loyalty-Focused']
    cities = ['Cairo', 'Alexandria', 'Giza', 'Sharm El Sheikh', 'Luxor', 'Aswan', 'Port Said', 'Tanta', 'Mansoura']
    
    customer_data = pd.DataFrame({
        'Customer ID': range(1, 101),
        'Segment': np.random.choice(segments, 100),
        'Purchase Frequency (Months)': np.random.randint(1, 12, 100),
        'Average Order Value (EGP)': np.random.randint(100, 1000, 100),
        'CLV (EGP)': np.random.randint(500, 5000, 100),
        'Churn Probability (%)': np.random.uniform(0, 100, 100),
        'Age': np.random.randint(18, 70, 100),
        'City': np.random.choice(cities, 100),
        'Profitability': np.random.choice(['Platinum', 'Gold', 'Iron'], 100)
    })

    return customer_data

# Initialize customer data
customer_data = generate_data()

# Streamlit page configuration
st.set_page_config(page_title="CRM Dashboard", layout="wide")

# Add custom CSS for header and subheader styling
st.markdown("""
    <style>
        .header {
            font-size: 40px;
            font-weight: bold;
            color: #1f77b4;
        }
        .subheader {
            font-size: 20px;
            color: #666;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("CRM Dashboard")
page = st.sidebar.radio("Choose a page", ["Customer Data Overview", "Customer Segmentation Report", "CLV Report", "Churn Report", "Purchase Frequency Report", "Loyalty Program Report", "Geographic Distribution"])

# Page Path navigation
st.markdown("""
    <style>
        .path {
            font-size: 14px;
            color: #999;
        }
    </style>
""", unsafe_allow_html=True)

# Add page path navigation
st.markdown('<div class="path">Home > CRM Dashboard > {}</div>'.format(page), unsafe_allow_html=True)

# Customer Data Overview
if page == "Customer Data Overview":
    st.markdown('<div class="header">Customer Data Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">A detailed overview of customer segmentation, CLV, purchase frequency, and churn probability.</div>', unsafe_allow_html=True)

    # Display the customer data
    st.dataframe(customer_data)

    # Histograms for Univariate Analysis
    st.markdown("### Univariate Analysis: Histograms")
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Purchase Frequency Distribution
    sns.histplot(customer_data['Purchase Frequency (Months)'], kde=True, color="skyblue", ax=ax[0])
    ax[0].set_title("Purchase Frequency Distribution")
    ax[0].set_xlabel("Purchase Frequency (Months)")
    ax[0].set_ylabel("Count")

    # CLV Distribution
    sns.histplot(customer_data['CLV (EGP)'], kde=True, color="orange", ax=ax[1])
    ax[1].set_title("CLV Distribution")
    ax[1].set_xlabel("Customer Lifetime Value (EGP)")
    ax[1].set_ylabel("Count")

    # Churn Probability Distribution
    sns.histplot(customer_data['Churn Probability (%)'], kde=True, color="green", ax=ax[2])
    ax[2].set_title("Churn Probability Distribution")
    ax[2].set_xlabel("Churn Probability (%)")
    ax[2].set_ylabel("Count")

    # Show Histograms
    st.pyplot(fig)

    # Scatter Plot for CLV vs. Purchase Frequency
    st.markdown("### Bivariate Analysis: CLV vs. Purchase Frequency")
    fig = px.scatter(customer_data, x='Purchase Frequency (Months)', y='CLV (EGP)',
                    color='Segment', title="CLV vs. Purchase Frequency",
                    labels={'Purchase Frequency (Months)': 'Purchase Frequency (Months)', 'CLV (EGP)': 'Customer Lifetime Value (EGP)'} )
    st.plotly_chart(fig, use_container_width=True)

    # Box Plot for CLV by Segment
    st.markdown("### Box Plot: CLV by Segment")
    fig = px.box(customer_data, x='Segment', y='CLV (EGP)', color='Segment',
                title="CLV by Segment",
                labels={'Segment': 'Customer Segment', 'CLV (EGP)': 'Customer Lifetime Value (EGP)'} )
    st.plotly_chart(fig, use_container_width=True)

    # Adding more visuals: Age by Segment
    col1, col2 = st.columns(2)

    # Age by Segment (Box Plot)
    with col1:
        st.markdown("### Age by Segment: Box Plot")
        fig = px.box(customer_data, x='Segment', y='Age', color='Segment',
                     title="Age Distribution by Segment")
        st.plotly_chart(fig, use_container_width=True)

    # Adding more visuals: Average Order Value (AOV) by Segment
    with col2:
        st.markdown("### Average Order Value (AOV) by Segment: Bar Chart")
        avg_aov_by_segment = customer_data.groupby('Segment')['Average Order Value (EGP)'].mean().reset_index()
        fig = px.bar(avg_aov_by_segment, x='Segment', y='Average Order Value (EGP)',
                     title="Average Order Value by Segment")
        st.plotly_chart(fig, use_container_width=True)

    # Adding more visuals: Churn Probability by Segment
    st.markdown("### Churn Probability by Segment: Box Plot")
    fig = px.box(customer_data, x='Segment', y='Churn Probability (%)', color='Segment',
                 title="Churn Probability Distribution by Segment")
    st.plotly_chart(fig, use_container_width=True)

    # Recommendations and Insights
    st.markdown("### Customer Data Overview Insights & Recommendations")
    st.write("""
    **Insights:**
    - Segments are relatively evenly distributed across the CLV and purchase frequency dimensions.
    - There is a clear distinction between the segments based on CLV, with 'Loyalty-Focused' customers having the highest average CLV.
    - Churn probability is spread across all segments, but 'Price-Sensitive' customers have a higher likelihood of churn.

    **Recommendations:**
    - **For Price-Sensitive Customers:**
        - Retention Strategies: Offer **loyalty programs**, **exclusive discounts**, or **bundled offers** to convert price-sensitive customers into loyal, higher-value customers.
        - Promotions: Regular, targeted promotional offers based on purchase history (e.g., discounts after a certain purchase frequency).
        - Referral Programs: Create **referral incentives** that reward customers for bringing in new customers, helping lower churn.
    - **For Loyalty-Focused Customers:**
        - Exclusive Deals: Provide **premium services** or **early access** to new products.
        - Customer Service: Ensure that **customer service** is exceptional for this segment, as they are more likely to recommend your brand and bring in new customers.
    - **For Innovation-Oriented Customers:**
        - Product Innovation: Continuously offer new, innovative products. As this group values innovation, introducing **new pasta shapes**, **flavors**, or **special edition packaging** will keep them engaged.
        - Personalized Engagement: Use customer data to **personalize offers** based on their purchase behavior, such as discounts for trying new products or personalized recommendations.
    """)


# Customer Segmentation Report
if page == "Customer Segmentation Report":
    st.markdown('<div class="header">Customer Segmentation Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Segmenting customers by purchase patterns, demographics, and preferences.</div>', unsafe_allow_html=True)

    # Segment analysis - Example: distribution of customer segments
    st.markdown('### Univariate Analysis')
    segment_dist = px.pie(customer_data, names='Segment', title='Customer Segment Distribution')
    st.plotly_chart(segment_dist, use_container_width=True)

    # Bivariate Analysis: Segment vs. CLV
    st.markdown('### Bivariate Analysis')
    segment_vs_clv = px.box(customer_data, x='Segment', y='CLV (EGP)', color='Segment', title='Segment vs CLV Distribution')
    st.plotly_chart(segment_vs_clv, use_container_width=True)

    # Customer Segmentation Heatmap
    st.markdown('### Customer Segmentation Heatmap')
    correlation_matrix = customer_data [['Purchase Frequency (Months)', 'Average Order Value (EGP)', 'CLV (EGP)', 'Churn Probability (%)', 'Age']].corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Recommendations for Improving Segmentation
    st.markdown("""
    **Segment Insights:**
    - **Price-Sensitive**: Focus on value-based offers, promotions, and creating strong referral programs.
    - **Innovation-Oriented**: Target this group with cutting-edge, limited-time offers and personalization through technology (e.g., mobile app offers).
    - **Loyalty-Focused**: Invest in loyalty programs, providing them exclusive offers and experiences to keep them engaged.
    """)

# Add more pages similarly: CLV, Churn, and Purchase Frequency Reports


# CLV Report
if page == "CLV Report":
    st.markdown('<div class="header">CLV Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Customer Lifetime Value (CLV) analysis by segment.</div>', unsafe_allow_html=True)

    # CLV analysis - Distribution
    st.markdown('### Univariate Analysis')
    clv_dist = px.histogram(customer_data, x='CLV (EGP)', nbins=20, title='CLV Distribution')
    st.plotly_chart(clv_dist, use_container_width=True)

    # Bivariate Analysis: CLV vs Purchase Frequency
    st.markdown('### Bivariate Analysis')
    clv_vs_purchase_freq = px.scatter(customer_data, x='Purchase Frequency (Months)', y='CLV (EGP)', color='Segment', title='CLV vs Purchase Frequency', color_discrete_sequence=['#66b3ff', '#99ff99', '#ffcc99'])
    st.plotly_chart(clv_vs_purchase_freq, use_container_width=True)

    # CLV by Segment (Bar Chart)
    st.markdown('### CLV by Segment')
    avg_clv_by_segment = customer_data.groupby('Segment')['CLV (EGP)'].mean().reset_index()
    clv_bar = px.bar(avg_clv_by_segment, x='Segment', y='CLV (EGP)', title='Average CLV by Segment')
    st.plotly_chart(clv_bar, use_container_width=True)

    # Download button for CLV data
    st.download_button("Download CLV Data", customer_data.to_csv(index=False), "clv_data.csv", key="download_clv_data")

    st.markdown("### CLV Report Insights & Recommendations")
    st.write("""
    **Insights:**
    - 'Loyalty-Focused' customers exhibit the highest CLV, suggesting they should be prioritized in long-term retention strategies.
    - 'Innovation-Oriented' customers have a moderate CLV but are more frequent buyers.

    **Recommendations:**
    - **Focus on Loyalty-Focused Customers:**
        - High-Value Engagement: Provide **loyalty programs** that reward repeat purchases, such as **tiered loyalty cards** that offer increasing benefits based on CLV.
        - Referral and Advocacy Programs: Since these customers are highly loyal, incentivize them with **referral bonuses** for bringing in new customers.
    - **Leverage Innovation-Oriented Customers:**
        - Frequent Purchase Incentives: To encourage repeat purchases, offer **early access** to new products or personalized subscription packages that make frequent buying more attractive.
        - Engagement through Feedback: Since this group is innovative, invite them to provide **feedback** on new product ideas, creating a more personalized and exclusive relationship.
    - **Use CLV Data to Forecast:**
        - Use the **CLV forecast** to prioritize customer service and marketing resources on the highest-value customers. Customers with high predicted CLV should receive **personalized attention** and exclusive offers.
    """)


# Churn Report
if page == "Churn Report":
    st.markdown('<div class="header">Churn Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Analyzing churn risk based on customer features.</div>', unsafe_allow_html=True)

    # Churn Prediction Model Evaluation
    st.markdown("### Churn Prediction Model Evaluation")
    X = customer_data[['Purchase Frequency (Months)', 'Average Order Value (EGP)', 'CLV (EGP)', 'Age']]
    y = customer_data['Churn Probability (%)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Convert continuous churn probabilities to binary (0 or 1) using a threshold (e.g., 50%)
    y_test_binary = (y_test > 50).astype(int)  # True churn (y_test > 50) becomes 1, else 0
    y_pred_binary = (y_pred > 50).astype(int)  # Predicted churn (y_pred > 50) becomes 1, else 0

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)
    fig = ff.create_annotated_heatmap(z=conf_matrix, colorscale='Blues', x=['Predicted No Churn', 'Predicted Churn'], y=['Actual No Churn', 'Actual Churn'])
    st.plotly_chart(fig, use_container_width=True)

    # At-Risk Customers by Segment
    st.markdown("### At-Risk Customers by Segment")
    customer_data['Churn Risk'] = (customer_data['Churn Probability (%)'] > 50).astype(int)
    at_risk_by_segment = customer_data.groupby(['Segment', 'Churn Risk'])['Customer ID'].count().unstack().fillna(0)
    churn_risk_bar = at_risk_by_segment.plot(kind='bar', stacked=True)
    st.pyplot(churn_risk_bar.figure)

    # Download button for churn data
    st.download_button("Download Churn Data", customer_data.to_csv(index=False), "churn_data.csv", key="download_churn_data")

    st.markdown("### Churn Report Insights & Recommendations")
    st.write("""
    **Insights:**
    - Churn risk is notably high among 'Price-Sensitive' customers.
    - 'Loyalty-Focused' customers show lower churn risk, reinforcing their value as long-term clients.
    - Churn model predicts higher risk for customers with lower purchase frequency and lower CLV.

    **Recommendations:**
    - **For Price-Sensitive Customers:**
        - Churn Prevention: To prevent churn, **target price-sensitive customers** with offers such as **loyalty points**, **exclusive discounts**, or **bundle offers**.
        - Retention Campaigns: Implement **automated email campaigns** that offer discounts or rewards for early renewal of purchases, or reminder emails for seasonal purchases.
    - **For Loyalty-Focused Customers:**
        - Exclusive Deals: Recognize the loyalty of these customers with **exclusive gifts** or **VIP experiences**. A personalized thank-you message on their anniversary or birthday could strengthen the relationship.
        - Enhanced Support: Offer **premium customer support** (e.g., priority helplines, early access to customer service) to ensure they feel valued.
    - **For Customers at Risk:**
        - Behavioral Retargeting: Use **predictive analytics** to identify at-risk customers and send them tailored offers (e.g., discounts or exclusive rewards) to re-engage them.
        - Personalized Re-engagement: Set up **automated workflows** that send a personalized **discount** or **exclusive offer** after a certain number of days of inactivity.
    """)

    

# Purchase Frequency Report
if page == "Purchase Frequency Report":
    st.markdown('<div class="header">Purchase Frequency Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Analyzing the frequency of customer purchases.</div>', unsafe_allow_html=True)

    # Univariate Analysis: Purchase Frequency Distribution
    st.markdown('### Univariate Analysis')
    purchase_freq_hist = px.histogram(customer_data, x='Purchase Frequency (Months)', nbins=12, title='Purchase Frequency Distribution')
    st.plotly_chart(purchase_freq_hist, use_container_width=True)

    # Bivariate Analysis: Purchase Frequency vs CLV
    st.markdown('### Bivariate Analysis')
    purchase_freq_vs_clv = px.scatter(customer_data, x='Purchase Frequency (Months)', y='CLV (EGP)', color='Segment', title='Purchase Frequency vs CLV')
    st.plotly_chart(purchase_freq_vs_clv, use_container_width=True)

    # Segmented Purchase Frequency (Bar Chart)
    st.markdown('### Segmented Purchase Frequency')
    purchase_freq_by_segment = customer_data.groupby('Segment')['Purchase Frequency (Months)'].mean().reset_index()
    purchase_freq_bar = px.bar(purchase_freq_by_segment, x='Segment', y='Purchase Frequency (Months)', title='Average Purchase Frequency by Segment')
    st.plotly_chart(purchase_freq_bar, use_container_width=True)

    # Download button for purchase frequency data
    st.download_button("Download Purchase Frequency Data", customer_data.to_csv(index=False), "purchase_freq_data.csv", key="download_purchase_freq_data")

    st.markdown("### Purchase Frequency Report Insights & Recommendations")
    st.write("""
    **Insights:**
    - Customers with higher purchase frequency also tend to have higher CLV.
    - 'Innovation-Oriented' customers demonstrate the highest purchase frequency, suggesting they are early adopters.

    **Recommendations:**
    - **For High-Frequency Purchasers:**
        - Subscription Programs: Offer **subscription services** (e.g., monthly deliveries of pasta) that incentivize high-frequency customers to commit to regular purchases.
        - Early Access or Pre-orders: Since these customers are frequent buyers, provide them with **early access** to new product releases or limited edition products.
    - **For Low-Frequency Purchasers:**
        - Incentivize Frequent Buying: Offer **time-limited promotions** or **discounted bundles** to encourage customers with lower purchase frequency to buy more often.
        - Personalized Discounts: Use **email marketing** or **SMS campaigns** with personalized discounts based on their purchase history to increase frequency.
    """)


    st.markdown("### Final Conclusion")
    st.write("""
    To sum up, Savola Foods can maximize customer value by:
    - **Segmenting customers effectively** and designing personalized strategies for each group. For instance, loyalty-focused customers should receive exclusive benefits and price-sensitive customers should be engaged through targeted promotions and loyalty programs.
    - **Increasing customer retention** through behavioral targeting, predictive churn models, and offering value through exclusive offers, loyalty programs, and personalized engagement.
    - **Improving CLV** by focusing on high-potential customers and offering them continuous value, whether through personalized offers or product innovation.
    - **Optimizing purchase frequency** by introducing subscription models, automated re-engagement campaigns, and reward-based incentives to encourage more frequent purchases.

    By leveraging machine learning models, CLV predictions, and churn analysis, Savola Foods can make **data-driven decisions** to enhance customer relationships, optimize marketing spend, and boost overall profitability. The actionable insights from the CRM analysis will guide Savola Foods to build long-lasting, profitable customer relationships that align with their strategic goals.
    """)

# Loyalty Program Report
if page == "Loyalty Program Report":
    st.markdown('<div class="header">Loyalty Program Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Analyzing customer participation in the loyalty program and its impact on CLV and churn.</div>', unsafe_allow_html=True)

    # Add Loyalty Program Column to the data
    customer_data['Loyalty Program'] = np.random.choice([True, False], size=len(customer_data))

    # Display loyalty program participation by segment
    loyalty_participation = customer_data.groupby(['Segment', 'Loyalty Program']).size().unstack().fillna(0)
    st.markdown("### Loyalty Program Participation by Segment")
    st.dataframe(loyalty_participation)

    # Bar chart of Loyalty Program participation by Segment
    st.markdown("### Loyalty Program Participation Distribution by Segment")
    fig = px.bar(loyalty_participation, x=loyalty_participation.index, y=loyalty_participation.columns, 
                 title="Loyalty Program Participation by Segment", 
                 labels={'Loyalty Program': 'Participation', 'Segment': 'Customer Segment'})
    st.plotly_chart(fig, use_container_width=True)

    # Correlation of Loyalty Program Participation with CLV
    st.markdown("### Impact of Loyalty Program on CLV")
    loyalty_clv = customer_data.groupby('Loyalty Program')['CLV (EGP)'].mean().reset_index()
    fig = px.bar(loyalty_clv, x='Loyalty Program', y='CLV (EGP)', 
                 title="Impact of Loyalty Program on CLV", 
                 labels={'Loyalty Program': 'Loyalty Program Participation', 'CLV (EGP)': 'Average CLV (EGP)'})
    st.plotly_chart(fig, use_container_width=True)

    # Box plot for CLV by Loyalty Program
    st.markdown("### CLV Distribution by Loyalty Program")
    fig = px.box(customer_data, x='Loyalty Program', y='CLV (EGP)', color='Loyalty Program', 
                 title="CLV Distribution by Loyalty Program")
    st.plotly_chart(fig, use_container_width=True)

    # Recommendations for Improving the Loyalty Program
    st.markdown("""
    **Insights:**
    - Customers who are part of the loyalty program generally have higher CLV compared to those who are not.
    - **Loyalty-Focused** customers tend to have higher loyalty program participation rates.

    **Recommendations:**
    - **For Non-Participants:** Introduce targeted marketing campaigns offering initial rewards for signing up, like **bonus points** or **exclusive discounts**.
    - **For High-Value Customers:** Offer **premium loyalty tiers** with exclusive perks like priority support, personalized services, and more rewards for frequent purchases.
    - **Gamification:** Consider adding **gamified elements** (e.g., milestones, rewards) to increase engagement in the loyalty program.
    """)

# Geographic Distribution Report
if page == "Geographic Distribution":
    st.markdown('<div class="header">Geographic Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Analyzing customer behavior across different regions in Egypt and understanding regional differences in key metrics.</div>', unsafe_allow_html=True)

    # Geographic Distribution: Pie chart by City
    st.markdown("### Customer Distribution by City")
    city_dist = customer_data['City'].value_counts().reset_index()
    city_dist.columns = ['City', 'Customer Count']
    fig = px.pie(city_dist, names='City', values='Customer Count', title='Customer Distribution by City')
    st.plotly_chart(fig, use_container_width=True)

    # CLV by City: Box plot
    st.markdown("### CLV Distribution by City")
    fig = px.box(customer_data, x='City', y='CLV (EGP)', color='City', 
                 title="CLV Distribution by City", 
                 labels={'City': 'City', 'CLV (EGP)': 'Customer Lifetime Value (EGP)'})
    st.plotly_chart(fig, use_container_width=True)

    # Churn Rate by City: Bar chart
    st.markdown("### Churn Probability by City")
    city_churn = customer_data.groupby('City')['Churn Probability (%)'].mean().reset_index()
    fig = px.bar(city_churn, x='City', y='Churn Probability (%)', 
                 title="Average Churn Probability by City", 
                 labels={'City': 'City', 'Churn Probability (%)': 'Churn Probability (%)'})
    st.plotly_chart(fig, use_container_width=True)

    # Recommendations for Regional Strategies
    st.markdown("""
    **Insights:**
    - **Cairo** and **Alexandria** have the highest concentration of customers, which could be targeted for more localized marketing campaigns.
    - **Churn Rates** are higher in **Port Said** and **Mansoura**, suggesting the need for tailored retention strategies in these regions.

    **Recommendations:**
    - **For High Churn Cities (e.g., Port Said, Mansoura):** 
        - Implement localized **retention strategies**, such as **targeted promotions** or special customer engagement programs.
        - Offer **regional discounts** or customized services to reduce churn.
    - **For High-Value Cities (e.g., Cairo, Alexandria):** 
        - Continue to nurture **loyalty programs** and introduce **premium services** for top-tier customers.
        - **Geo-targeted advertising** could help focus efforts on these high-potential areas for increasing CLV.
    """)


