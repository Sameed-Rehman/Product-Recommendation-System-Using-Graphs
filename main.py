import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import numpy as np

# Step 1: Generate a larger dataset
random.seed(42)

# Parameters
num_orders = 5
products = [chr(i) for i in range(65, 91)]  # Products A to Z
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 2, 1)

# Generate random orders
data = {
    "Order ID": [],
    "Product": [],
    "Quantity": [],
    "Price": [],
    "Order Date": []
}

for order_id in range(1, num_orders + 1):
    num_products = random.randint(2, 6)  # Each order has 2-6 products
    selected_products = random.sample(products, num_products)
    order_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    
    for product in selected_products:
        data["Order ID"].append(order_id)
        data["Product"].append(product)
        data["Quantity"].append(random.randint(1, 5))  # Random quantity (1-5)
        data["Price"].append(random.randint(10, 100))  # Random price (10-100)
        data["Order Date"].append(order_date)

# Create the DataFrame
df = pd.DataFrame(data)

# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Step 2: Calculate RFM scores
current_date = end_date + timedelta(days=1)

rfm = df.groupby('Product').agg(
    Recency=('Order Date', lambda x: (current_date - x.max()).days),
    Frequency=('Product', 'count'),
    Monetary=('Price', lambda x: (x * df.loc[x.index, 'Quantity']).sum())
)

# Normalize RFM scores
rfm['Recency'] += np.random.uniform(0, 0.01, size=len(rfm))
rfm['Frequency'] += np.random.uniform(0, 0.01, size=len(rfm))
rfm['Monetary'] += np.random.uniform(0, 0.01, size=len(rfm))
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=range(5, 0, -1)).astype(int)
rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=5, labels=range(1, 6)).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=range(1, 6)).astype(int)

# Combine RFM scores into a weight
rfm['Weight'] = 0.4 * rfm['R_Score'] + 0.3 * rfm['F_Score'] + 0.3 * rfm['M_Score']

# Step 3: Create the RFM-weighted graph
G = nx.DiGraph()

# Add nodes (products)
for product in rfm.index:
    G.add_node(product)

# Create edges based on co-occurrence in orders
order_groups = df.groupby('Order ID')['Product'].apply(list)

for order in order_groups:
    for i, product_a in enumerate(order):
        for product_b in order:
            if product_a != product_b:  # Avoid self-loops
                weight = rfm.loc[product_b, 'Weight'] if product_b in rfm.index else 0
                if G.has_edge(product_a, product_b):
                    G[product_a][product_b]['weight'] += weight  # Increment existing weight
                else:
                    G.add_edge(product_a, product_b, weight=weight)  # Add new edge

# Step 4: Visualize the RFM table
plt.figure(figsize=(12, 6))
plt.axis('off')
plt.title("RFM Table", fontsize=18, pad=20)
table = plt.table(
    cellText=rfm.reset_index().round(2).values,
    colLabels=["Product", "Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score", "Weight"],
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(rfm.columns) + 1)))
plt.show()

# Step 5: Visualize the graph
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels={k: round(v, 2) for k, v in edge_labels.items()}, font_size=8)

plt.title("RFM-Weighted Product Recommendation Graph", fontsize=16)
plt.show()

# Step 6: Recommend products
while True:
    user_product = input("\nEnter a product to get recommendations (or type 'exit' to quit): ").upper()
    if user_product == 'EXIT':
        print("Exiting recommendation system. Goodbye!")
        break
    if user_product not in G.nodes:
        print(f"Product '{user_product}' does not exist in the graph. Please try again.")
        continue
    
    # Get recommendations based on outgoing edges
    recommendations = sorted(
        G[user_product].items(), 
        key=lambda x: x[1]['weight'], 
        reverse=True
    )
    if recommendations:
        print(f"Recommended products for '{user_product}':")
        for rec_product, data in recommendations:
            print(f"  - {rec_product} (Weight: {round(data['weight'], 2)})")
    else:
        print(f"No recommendations available for '{user_product}'.")
