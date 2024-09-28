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

# Step 4: Analyze Sales Based on Opening Status
if 'Sales' in train_data.columns and 'Open' in train_data.columns:
    # Step 5: Create a summary of sales based on whether stores were open
    sales_summary = train_data.groupby('Open')['Sales'].mean().reset_index()

    # Step 6: Visualize the Impact of Store Opening on Sales
    plt.figure(figsize=(12, 6))
    sns.barplot(data=sales_summary, x='Open', y='Sales', palette='viridis')
    plt.title('Average Sales Based on Store Opening Status')
    plt.xlabel('Store Open (0 = Closed, 1 = Open)')
    plt.ylabel('Average Sales')
    plt.xticks(ticks=[0, 1], labels=['Closed', 'Open'])
    
    # Save the store opening trends plot as JPG
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\store_opening_trends.jpg', format='jpg')
    plt.close()  # Close the figure to save memory

    # Step 7: Discuss the Findings
    print("\n**Store Opening Trends Analysis:**")
    print("The analysis reveals that stores that are open generate significantly higher average sales compared to when they are closed. "
          "This insight indicates that optimizing store schedules to ensure maximum operating hours could lead to increased revenue generation.")

# Step 8: Prepare the Test Data for Predictions
if 'Date' in test_data.columns:
    test_data['Date'] = pd.to_datetime(test_data['Date'], format='%Y-%m-%d')

# (Optional) Further processing or model predictions can be added here
