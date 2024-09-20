import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Training Data with dtype specifications
train_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\train.csv', 
                          dtype={'StateHoliday': 'str', 'SchoolHoliday': 'int'}, 
                          delimiter=',')  # Use ',' for comma-separated data

# Step 2: Load the Test Data with dtype specifications
test_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\test.csv', 
                         dtype={'StateHoliday': 'str'}, 
                         delimiter=',')

# Step 3: Load Store Data
store_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\store.csv', 
                          delimiter=',')

# Step 4: Merge Store Data with Training Data
merged_data = pd.merge(train_data, store_data, on='Store')

# Step 5: Prepare Data for Visualization
# 1. Daily Sales Over Time
daily_sales = merged_data.groupby('Date')['Sales'].sum().reset_index()

# 2. Average Sales by Assortment Type
assortment_sales = merged_data.groupby('Assortment')['Sales'].mean().reset_index()

# 3. Average Sales by Competition Distance
merged_data['Distance_Category'] = pd.cut(merged_data['CompetitionDistance'], 
                                            bins=[0, 500, 1000, 5000, 15000], 
                                            labels=['Very Close', 'Close', 'Moderate', 'Far'])
competition_sales = merged_data.groupby('Distance_Category')['Sales'].mean().reset_index()

# Step 6: Create 2x2 Grid for Visualization
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Total Sales Over Time
sns.lineplot(data=daily_sales, x='Date', y='Sales', marker='o', ax=axs[0, 0], color='blue')
axs[0, 0].set_title('Total Sales Over Time')
axs[0, 0].set_xlabel('Date')
axs[0, 0].set_ylabel('Total Sales')
axs[0, 0].tick_params(axis='x', rotation=45)
axs[0, 0].grid()

# Plot 2: Average Sales by Assortment Type
sns.barplot(data=assortment_sales, x='Assortment', y='Sales', ax=axs[0, 1], palette='viridis')
axs[0, 1].set_title('Average Sales by Assortment Type')
axs[0, 1].set_xlabel('Assortment Type')
axs[0, 1].set_ylabel('Average Sales')

# Plot 3: Average Sales by Competition Distance
sns.barplot(data=competition_sales, x='Distance_Category', y='Sales', ax=axs[1, 0], palette='viridis')
axs[1, 0].set_title('Average Sales by Competition Distance')
axs[1, 0].set_xlabel('Competition Distance Category')
axs[1, 0].set_ylabel('Average Sales')

# Remove empty subplot
axs[1, 1].axis('off')

# Adjust layout
plt.tight_layout()

# Save the 2x2 grid visualization as JPG
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\combined_visualizations.jpg', format='jpg')
plt.close()

# Step 7: Prepare the Test Data for Predictions
if 'Date' in test_data.columns:
    test_data['Date'] = pd.to_datetime(test_data['Date'], format='%Y-%m-%d')

# Summary output for stakeholders
print("\n**Visualization Summary:**")
print("1. Total sales over time plotted to observe trends.")
print("2. Average sales by assortment type provided to highlight product variety impact.")
print("3. Average sales by competition distance included to understand local dynamics.")
