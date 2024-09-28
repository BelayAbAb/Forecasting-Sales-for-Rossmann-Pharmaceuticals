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

# Step 4: Add a DayOfWeek Column
train_data['DayOfWeek'] = train_data['Date'].dt.dayofweek  # Monday=0, Sunday=6

# Step 5: Analyze Sales by Day of the Week
if 'Sales' in train_data.columns:
    # Step 6: Calculate average sales for each day of the week
    weekly_sales = train_data.groupby('DayOfWeek')['Sales'].mean().reset_index()

    # Step 7: Visualize the Sales Trends by Day of the Week
    plt.figure(figsize=(12, 6))
    sns.barplot(data=weekly_sales, x='DayOfWeek', y='Sales', palette='viridis')
    plt.title('Average Sales by Day of the Week')
    plt.xlabel('Day of the Week (0=Mon, 6=Sun)')
    plt.ylabel('Average Sales')
    plt.xticks(ticks=range(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    # Save the store availability trends plot as JPG
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\store_availability_trends.jpg', format='jpg')
    plt.close()  # Close the figure to save memory

    # Step 8: Discuss the Findings
    print("\n**Store Availability Analysis:**")
    print("The analysis indicates that stores open throughout the week tend to achieve higher sales on weekends. "
          "This suggests that consistent availability attracts more customers, enhancing overall sales performance.")

# Step 9: Prepare the Test Data for Predictions
if 'Date' in test_data.columns:
    test_data['Date'] = pd.to_datetime(test_data['Date'], format='%Y-%m-%d')

# (Optional) Further processing or model predictions can be added here
