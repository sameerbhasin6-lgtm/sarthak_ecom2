import streamlit as st
import pandas as pd
from scipy.optimize import minimize
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Static AI Pricing Agent", layout="wide")
st.title("🤖 AI Pricing Agent (Static Data)")

# --- STEP 1: HARDCODED DATA (Based on your Excel File) ---
# This dictionary contains the exact data from "Provide the complete required data set..."
data = {
    'Day': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 
            5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 
            9, 9, 9, 9, 9, 10, 10, 10, 10, 10], 
    'Product_ID': ['P1', 'P2', 'P3', 'P4', 'P5', 'P1', 'P2', 'P3', 'P4', 'P5', 
                   'P1', 'P2', 'P3', 'P4', 'P5', 'P1', 'P2', 'P3', 'P4', 'P5', 
                   'P1', 'P2', 'P3', 'P4', 'P5', 'P1', 'P2', 'P3', 'P4', 'P5', 
                   'P1', 'P2', 'P3', 'P4', 'P5', 'P1', 'P2', 'P3', 'P4', 'P5', 
                   'P1', 'P2', 'P3', 'P4', 'P5', 'P1', 'P2', 'P3', 'P4', 'P5'], 
    'Cost': [4000, 3500, 800, 2500, 1200, 4000, 3500, 800, 2500, 1200, 
             4000, 3500, 800, 2500, 1200, 4000, 3500, 800, 2500, 1200, 
             4000, 3500, 800, 2500, 1200, 4000, 3500, 800, 2500, 1200, 
             4000, 3500, 800, 2500, 1200, 4000, 3500, 800, 2500, 1200, 
             4000, 3500, 800, 2500, 1200, 4000, 3500, 800, 2500, 1200], 
    'Inventory': [80, 120, 300, 90, 150, 75, 115, 290, 85, 145, 
                  70, 110, 270, 80, 140, 65, 105, 240, 75, 135, 
                  60, 100, 220, 70, 130, 55, 95, 200, 65, 125, 
                  50, 90, 180, 60, 120, 45, 85, 160, 55, 115, 
                  40, 80, 140, 50, 110, 35, 75, 120, 45, 105], 
    'Comp_Price': [5200, 4300, 1200, 3100, 1700, 5300, 4250, 1250, 3150, 1750, 
                   5100, 4400, 1150, 3050, 1650, 5400, 4500, 1300, 3200, 1800, 
                   5250, 4350, 1225, 3125, 1725, 5350, 4450, 1275, 3175, 1775, 
                   5150, 4300, 1175, 3075, 1675, 5450, 4550, 1325, 3225, 1825, 
                   5200, 4250, 1200, 3100, 1700, 5300, 4350, 1250, 3150, 1750], 
    'Hist_Demand': [18, 25, 45, 20, 30, 17, 26, 40, 19, 32, 
                    20, 22, 50, 21, 28, 16, 28, 35, 18, 35, 
                    19, 23, 42, 22, 29, 15, 27, 38, 17, 31, 
                    21, 24, 48, 20, 27, 14, 29, 33, 16, 34, 
                    18, 25, 45, 19, 30, 17, 26, 40, 18, 32], 
    'Hist_Price': [5000, 4200, 1150, 3000, 1600, 5000, 4200, 1150, 3000, 1600, 
                   5000, 4200, 1150, 3000, 1600, 5000, 4200, 1150, 3000, 1600, 
                   5000, 4200, 1150, 3000, 1600, 5000, 4200, 1150, 3000, 1600, 
                   5000, 4200, 1150, 3000, 1600, 5000, 4200, 1150, 3000, 1600, 
                   5000, 4200, 1150, 3000, 1600, 5000, 4200, 1150, 3000, 1600]
}

df = pd.DataFrame(data)

# --- STEP 2: METADATA & ELASTICITY MAPPING ---
product_map = {
    'P1': {'Cat': 'Cricket',   'Elas': -1.5, 'Name': 'English Willow Bat'},
    'P2': {'Cat': 'Football',  'Elas': -2.0, 'Name': 'Pro Match Ball'},
    'P3': {'Cat': 'Swimming',  'Elas': -2.5, 'Name': 'Anti-Fog Goggles'},
    'P4': {'Cat': 'Badminton', 'Elas': -1.8, 'Name': 'Carbon Racket'},
    'P5': {'Cat': 'Fitness',   'Elas': -1.2, 'Name': 'Dumbbell Set'}
}

def get_meta(pid, field):
    return product_map.get(pid, {'Cat': 'General', 'Elas': -1.5, 'Name': pid})[field]

df['Category'] = df['Product_ID'].apply(lambda x: get_meta(x, 'Cat'))
df['Product_Name'] = df['Product_ID'].apply(lambda x: get_meta(x, 'Name'))
df['Elasticity'] = df['Product_ID'].apply(lambda x: get_meta(x, 'Elas'))

# --- STEP 3: OPTIMIZATION ENGINE ---
def optimize_price(row):
    cost = row['Cost']
    comp_price = row['Comp_Price']
    inventory = row['Inventory']
    hist_demand = row['Hist_Demand']
    hist_avg_price = row['Hist_Price']
    elasticity = row['Elasticity']

    # Objective: Maximize Profit
    def objective(price):
        p = price[0]
        # Demand Logic: New_Demand = Hist_Demand * (1 + Elasticity * %Change_in_Price)
        pct_change = (p - hist_avg_price) / hist_avg_price
        pred_demand = hist_demand * (1 + elasticity * pct_change)
        
        profit = (p - cost) * pred_demand
        return -profit # Negate for minimization

    # Constraints
    cons = [
        {'type': 'ineq', 'fun': lambda p: p[0] - (cost * 1.10)}, # Price >= Cost + 10%
        {'type': 'ineq', 'fun': lambda p: (comp_price * 1.25) - p[0]}, # Price <= Comp * 1.25
        # Inventory: Sales <= Inventory
        {'type': 'ineq', 'fun': lambda p: inventory - (hist_demand * (1 + elasticity * ((p[0] - hist_avg_price) / hist_avg_price)))}
    ]

    # Run Solver
    result = minimize(objective, [comp_price], constraints=cons, bounds=[(cost, comp_price*2)])
    
    opt_price = result.x[0]
    pct_change_opt = (opt_price - hist_avg_price) / hist_avg_price
    opt_demand = hist_demand * (1 + elasticity * pct_change_opt)
    opt_profit = (opt_price - cost) * opt_demand
    
    base_profit = (hist_avg_price - cost) * hist_demand

    return pd.Series([opt_price, opt_demand, opt_profit, base_profit], 
                     index=['Rec_Price', 'Pred_Sales', 'Proj_Profit', 'Base_Profit'])

# --- STEP 4: RUN & DISPLAY DASHBOARD ---
if st.button("🚀 Run AI Optimization Now"):
    with st.spinner("Analyzing static dataset..."):
        results = df.apply(optimize_price, axis=1)
        final_df = pd.concat([df, results], axis=1)

        # Summary Metrics
        total_base = final_df['Base_Profit'].sum()
        total_proj = final_df['Proj_Profit'].sum()
        uplift = ((total_proj - total_base) / total_base) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Historical Profit", f"₹{total_base:,.0f}")
        col2.metric("AI Optimized Profit", f"₹{total_proj:,.0f}")
        col3.metric("Profit Uplift", f"🚀 {uplift:.2f}%")

        # Visualization
        st.subheader("Pricing Strategy: English Willow Bat (P1)")
        p1_data = final_df[final_df['Product_ID'] == 'P1']
        fig = px.line(p1_data, x='Day', y=['Hist_Price', 'Comp_Price', 'Rec_Price'],
                      title="AI Price Recommendation vs Competitor",
                      markers=True)
        st.plotly_chart(fig, use_container_width=True)

        # Detailed Table
        st.subheader("Full Optimization Results")
        st.dataframe(final_df[['Day', 'Product_ID', 'Category', 'Cost', 'Comp_Price', 'Rec_Price', 'Pred_Sales', 'Proj_Profit']].style.format({
            'Cost': '₹{:.0f}', 'Comp_Price': '₹{:.0f}', 'Rec_Price': '₹{:.2f}', 'Pred_Sales': '{:.1f}', 'Proj_Profit': '₹{:.0f}'
        }))
else:
    st.info("Click the button above to run the optimization on your static dataset.")
    st.write("### Preview of Loaded Data")
    st.dataframe(df.head())