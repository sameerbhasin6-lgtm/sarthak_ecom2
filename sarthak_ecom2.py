import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hustler AI Pricing Manager", layout="wide")
st.title("🤖 Hustler Store: AI Pricing Agent")
st.markdown("### 🛒 Dynamic Pricing for New Product Lineup")

# --- STEP 1: NEW STATIC DATASET (7 Products x 10 Days) ---
# Generated specifically for the new Hustler product list
data = {
    'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 7,
    'Product_ID': 
        ['P1'] * 10 + ['P2'] * 10 + ['P3'] * 10 + ['P4'] * 10 + 
        ['P5'] * 10 + ['P6'] * 10 + ['P7'] * 10,
    'Product_Name': 
        ['Wardrobe Organizer (4pcs)'] * 10 + 
        ['Adj. Hand Gripper (100kg)'] * 10 + 
        ['PVC Dumbbell Set'] * 10 + 
        ['Skipping Rope'] * 10 + 
        ['Tumbler Bottle (400ml)'] * 10 + 
        ['Warrior Tumbler (40oz)'] * 10 + 
        ['Gym Gloves'] * 10,
    'Cost': 
        [450]*10 + [350]*10 + [1200]*10 + [150]*10 + [350]*10 + [650]*10 + [250]*10,
    # Simulating slight daily competitor price fluctuations
    'Comp_Price': 
        [799, 790, 810, 800, 795, 780, 805, 799, 815, 790] +  # P1
        [599, 580, 600, 590, 595, 585, 610, 599, 575, 590] +  # P2
        [2499, 2450, 2500, 2480, 2490, 2400, 2550, 2499, 2420, 2480] + # P3
        [299, 280, 310, 290, 295, 285, 305, 299, 275, 290] +  # P4
        [699, 680, 710, 690, 695, 685, 705, 699, 675, 690] +  # P5
        [1299, 1250, 1300, 1280, 1290, 1240, 1310, 1299, 1260, 1280] + # P6
        [499, 480, 510, 490, 495, 485, 505, 499, 475, 490],   # P7
    'Inventory': 
        [100, 95, 90, 85, 80, 75, 70, 65, 60, 55] +   # P1
        [150, 145, 140, 135, 130, 125, 120, 115, 110, 105] + # P2
        [50, 48, 46, 44, 42, 40, 38, 36, 34, 32] +    # P3
        [200, 190, 180, 170, 160, 150, 140, 130, 120, 110] + # P4
        [120, 115, 110, 105, 100, 95, 90, 85, 80, 75] + # P5
        [80, 76, 72, 68, 64, 60, 56, 52, 48, 44] +    # P6
        [180, 170, 160, 150, 140, 130, 120, 110, 100, 90],   # P7
    'Hist_Demand': 
        [15, 16, 14, 15, 17, 18, 14, 15, 16, 15] + 
        [25, 28, 22, 25, 26, 30, 24, 25, 29, 25] + 
        [8, 9, 7, 8, 9, 10, 7, 8, 9, 8] + 
        [40, 45, 38, 40, 42, 48, 39, 40, 46, 40] + 
        [20, 22, 18, 20, 21, 23, 19, 20, 24, 20] + 
        [12, 13, 11, 12, 14, 15, 12, 12, 14, 12] + 
        [35, 38, 32, 35, 36, 40, 33, 35, 39, 35],
    'Elasticity': 
        [-1.8] * 10 + # Organizer (Elastic)
        [-1.5] * 10 + # Gripper
        [-1.2] * 10 + # Dumbbells (Less elastic, heavy shipping)
        [-2.5] * 10 + # Rope (Cheap, very elastic)
        [-2.0] * 10 + # Tumbler 400ml
        [-1.8] * 10 + # Tumbler 40oz (Trend item)
        [-2.2] * 10   # Gloves
}

df = pd.DataFrame(data)

# --- STEP 2: OPTIMIZATION ENGINE ---
def optimize_price(row):
    cost = row['Cost']
    comp_price = row['Comp_Price']
    inventory = row['Inventory']
    base_demand = row['Hist_Demand']
    elasticity = row['Elasticity']
    
    # We assume "Historical Price" was roughly the Competitor Price for the baseline
    hist_avg_price = comp_price 

    # Objective: Maximize Profit
    def objective(price):
        p = price[0]
        # Demand Formula
        pct_change = (p - hist_avg_price) / hist_avg_price
        pred_demand = base_demand * (1 + elasticity * pct_change)
        
        profit = (p - cost) * pred_demand
        return -profit

    # Constraints
    cons = [
        {'type': 'ineq', 'fun': lambda p: p[0] - (cost * 1.15)}, # Min Margin 15%
        {'type': 'ineq', 'fun': lambda p: (comp_price * 1.30) - p[0]}, # Max Cap 30% above rival
        {'type': 'ineq', 'fun': lambda p: inventory - (base_demand * (1 + elasticity * ((p[0] - hist_avg_price) / hist_avg_price)))}
    ]

    result = minimize(objective, [comp_price], constraints=cons, bounds=[(cost, comp_price*2)])
    
    opt_price = result.x[0]
    pct_change_opt = (opt_price - hist_avg_price) / hist_avg_price
    opt_demand = base_demand * (1 + elasticity * pct_change_opt)
    
    # Financials
    opt_revenue = opt_price * opt_demand
    opt_profit = (opt_price - cost) * opt_demand
    
    base_revenue = comp_price * base_demand
    base_profit = (comp_price - cost) * base_demand

    return pd.Series([opt_price, opt_demand, opt_revenue, opt_profit, base_revenue, base_profit], 
                     index=['Rec_Price', 'Pred_Sales', 'Opt_Revenue', 'Opt_Profit', 'Base_Revenue', 'Base_Profit'])

# --- STEP 3: DASHBOARD EXECUTION ---
if st.button("🚀 Run AI Optimization for New List"):
    with st.spinner("Analyzing market data for 7 products..."):
        results = df.apply(optimize_price, axis=1)
        final_df = pd.concat([df, results], axis=1)

        # 1. KPI Metrics
        total_base_rev = final_df['Base_Revenue'].sum()
        total_opt_rev = final_df['Opt_Revenue'].sum()
        rev_uplift = ((total_opt_rev - total_base_rev) / total_base_rev) * 100
        
        total_base_prof = final_df['Base_Profit'].sum()
        total_opt_prof = final_df['Opt_Profit'].sum()
        prof_uplift = ((total_opt_prof - total_base_prof) / total_base_prof) * 100

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Revenue (AI)", f"₹{total_opt_rev:,.0f}", f"↑ {rev_uplift:.1f}%")
        kpi2.metric("Total Profit (AI)", f"₹{total_opt_prof:,.0f}", f"↑ {prof_uplift:.1f}%")
        kpi3.metric("Baseline Revenue", f"₹{total_base_rev:,.0f}")
        kpi4.metric("Baseline Profit", f"₹{total_base_prof:,.0f}")

        st.divider()

        # 2. GRAPHS ROW 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Revenue Comparison")
            # Aggregating by Product Name
            prod_group = final_df.groupby('Product_Name')[['Base_Revenue', 'Opt_Revenue']].sum().reset_index()
            prod_melt = prod_group.melt(id_vars='Product_Name', var_name='Scenario', value_name='Revenue')
            
            fig_rev = px.bar(prod_melt, x='Product_Name', y='Revenue', color='Scenario', barmode='group',
                             color_discrete_map={'Base_Revenue': '#95a5a6', 'Opt_Revenue': '#2ecc71'},
                             title="Revenue Uplift by Product")
            st.plotly_chart(fig_rev, use_container_width=True)

        with col2:
            st.subheader("📦 Inventory vs. Predicted Sales")
            # Check for stockout risks (Day 1 snapshot or Average)
            inv_data = final_df.groupby('Product_Name')[['Inventory', 'Pred_Sales']].mean().reset_index()
            fig_inv = px.bar(inv_data, x='Product_Name', y=['Inventory', 'Pred_Sales'], barmode='overlay',
                             title="Average Stock vs. Daily Demand",
                             color_discrete_map={'Inventory': '#3498db', 'Pred_Sales': '#e74c3c'})
            st.plotly_chart(fig_inv, use_container_width=True)

        # 3. GRAPHS ROW 2
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("📉 Demand Elasticity Curve")
            # Scatter plot: % Price Change vs % Demand Change
            final_df['Pct_Price_Change'] = (final_df['Rec_Price'] - final_df['Comp_Price']) / final_df['Comp_Price']
            final_df['Pct_Demand_Change'] = (final_df['Pred_Sales'] - final_df['Hist_Demand']) / final_df['Hist_Demand']
            
            fig_elast = px.scatter(final_df, x='Pct_Price_Change', y='Pct_Demand_Change', color='Product_Name',
                                   title="Price Sensitivity (Elasticity Analysis)",
                                   labels={'Pct_Price_Change': 'Price Increase %', 'Pct_Demand_Change': 'Demand Drop %'})
            st.plotly_chart(fig_elast, use_container_width=True)

        with col4:
            st.subheader("🏷️ Price Strategy: Competitor vs AI")
            # Box plot to show price ranges
            price_melt = final_df.melt(id_vars='Product_Name', value_vars=['Comp_Price', 'Rec_Price'], 
                                       var_name='Type', value_name='Price')
            fig_box = px.box(price_melt, x='Product_Name', y='Price', color='Type',
                             title="Pricing Spread Distribution")
            st.plotly_chart(fig_box, use_container_width=True)

        # 4. DATA TABLE
        st.subheader("📋 Detailed AI Recommendations")
        st.dataframe(final_df[['Product_Name', 'Cost', 'Comp_Price', 'Rec_Price', 'Hist_Demand', 'Pred_Sales', 'Opt_Profit']].head(20).style.format({
            'Cost': '₹{:.0f}', 'Comp_Price': '₹{:.0f}', 'Rec_Price': '₹{:.2f}', 
            'Hist_Demand': '{:.1f}', 'Pred_Sales': '{:.1f}', 'Opt_Profit': '₹{:.0f}'
        }))

else:
    st.info("Click the button above to generate the new data and run the optimization.")
    st.write("### New Product List Preview")
    st.table(pd.DataFrame(data).head(7)[['Product_ID', 'Product_Name', 'Cost', 'Comp_Price']])
