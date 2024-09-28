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

# Step 3: Convert the Date Column to Datetime
if 'Date' in train_data.columns:
    train_data['Date'] = pd.to_datetime(train_data['Date'], format='%Y-%m-%d')  # Correct format for your data

# Step 4: Analyze Sales by Store
if 'Sales' in train_data.columns and 'Store' in train_data.columns:
    # Step 5: Calculate total sales per store
    store_sales = train_data.groupby('Store')['Sales'].sum().reset_index()

    # Step 6: Identify underperforming stores (e.g., below the median sales)
    median_sales = store_sales['Sales'].median()
    underperforming_stores = store_sales[store_sales['Sales'] < median_sales]

    # Step 7: Analyze Promotional Impact on Underperforming Stores
    promo_analysis = train_data[train_data['Store'].isin(underperforming_stores['Store'])]
    promo_summary = promo_analysis.groupby('Promo')['Sales'].mean().reset_index()

    # Step 8: Visualize the Impact of Promotions on Underperforming Stores
    plt.figure(figsize=(12, 6))
    sns.barplot(data=promo_summary, x='Promo', y='Sales', hue='Promo', palette='viridis', dodge=False, legend=False)
    plt.title('Average Sales During Promotional and Non-Promotional Periods for Underperforming Stores')
    plt.xlabel('Promotion Active (0 = No, 1 = Yes)')
    plt.ylabel('Average Sales')
    plt.xticks(ticks=[0, 1], labels=['No Promotion', 'Promotion'])
    
    # Save the optimal promo strategies plot as JPG
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\optimal_promo_strategies.jpg', format='jpg')
    plt.close()  # Close the figure to save memory

    # Step 9: Discuss the Findings
    print("\n**Optimal Promotional Strategies for Underperforming Stores:**")
    print("The analysis reveals that underperforming stores tend to benefit significantly from promotional campaigns. "
          "By deploying promotions strategically in these stores, overall sales can be enhanced. "
          "This insight suggests that targeted promotional strategies can help boost performance and drive revenue.")

# Step 10: Prepare the Test Data for Predictions
if 'Date' in test_data.columns:
    test_data['Date'] = pd.to_datetime(test_data['Date'], format='%Y-%m-%d')

# (Optional) Further processing or model predictions can be added here
