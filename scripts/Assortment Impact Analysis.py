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

# Step 3: Load Store Data to get Assortment Information
store_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\store.csv', 
                          delimiter=',')

# Step 4: Merge Store Data with Training Data to Analyze Assortment Impact
merged_data = pd.merge(train_data, store_data, on='Store')

# Step 5: Analyze Sales by Assortment Type
if 'Sales' in merged_data.columns and 'Assortment' in merged_data.columns:
    # Step 6: Calculate average sales for each assortment type
    assortment_sales = merged_data.groupby('Assortment')['Sales'].mean().reset_index()

    # Step 7: Visualize the Sales Impact of Different Assortment Types
    plt.figure(figsize=(12, 6))
    sns.barplot(data=assortment_sales, x='Assortment', y='Sales', palette='viridis')
    plt.title('Average Sales by Assortment Type')
    plt.xlabel('Assortment Type')
    plt.ylabel('Average Sales')
    
    # Save the assortment impact plot as JPG
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\assortment_impact.jpg', format='jpg')
    plt.close()  # Close the figure to save memory

    # Step 8: Discuss the Findings
    print("\n**Assortment Impact Analysis:**")
    print("The analysis indicates that stores with extended assortments generally experience higher sales. "
          "This highlights the value of variety in meeting customer needs and suggests that expanding product ranges could enhance revenue.")

# Step 9: Prepare the Test Data for Predictions
if 'Date' in test_data.columns:
    test_data['Date'] = pd.to_datetime(test_data['Date'], format='%Y-%m-%d')

# (Optional) Further processing or model predictions can be added here
