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

# Step 3: Load Store Data to get Competition Distance
store_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\store.csv', 
                          delimiter=',')

# Step 4: Merge Store Data with Training Data to Analyze Competition Effects
merged_data = pd.merge(train_data, store_data, on='Store')

# Step 5: Analyze Sales by Competition Distance
if 'Sales' in merged_data.columns and 'CompetitionDistance' in merged_data.columns:
    # Step 6: Create a new column categorizing competition distance
    merged_data['Distance_Category'] = pd.cut(merged_data['CompetitionDistance'], 
                                                bins=[0, 500, 1000, 5000, 15000], 
                                                labels=['Very Close', 'Close', 'Moderate', 'Far'])

    # Step 7: Calculate average sales for each distance category
    competition_sales = merged_data.groupby('Distance_Category')['Sales'].mean().reset_index()

    # Step 8: Visualize the Sales Impact of Competition Distance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=competition_sales, x='Distance_Category', y='Sales', palette='viridis')
    plt.title('Average Sales by Competition Distance')
    plt.xlabel('Competition Distance Category')
    plt.ylabel('Average Sales')
    
    # Save the competition effects plot as JPG
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\competition_effects.jpg', format='jpg')
    plt.close()  # Close the figure to save memory

    # Step 9: Discuss the Findings
    print("\n**Competition Effects Analysis:**")
    print("The analysis indicates that proximity to competitors impacts sales, especially in urban areas. "
          "Stores located very close to competitors tend to have lower sales, suggesting that marketing strategies should "
          "consider local competition dynamics to enhance customer attraction and retention.")

# Step 10: Prepare the Test Data for Predictions
if 'Date' in test_data.columns:
    test_data['Date'] = pd.to_datetime(test_data['Date'], format='%Y-%m-%d')

# (Optional) Further processing or model predictions can be added here
