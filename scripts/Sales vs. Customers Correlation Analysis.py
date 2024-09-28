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

# Step 4: Analyze Sales and Customers Data
if 'Sales' in train_data.columns and 'Customers' in train_data.columns:
    # Step 5: Create a summary DataFrame
    summary = train_data.groupby('Date')[['Sales', 'Customers']].sum().reset_index()

    # Step 6: Calculate Correlation
    correlation = summary['Sales'].corr(summary['Customers'])
    print(f"Correlation between Sales and Customers: {correlation:.2f}")

    # Step 7: Visualize the Correlation
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=summary, x='Customers', y='Sales', alpha=0.6)
    plt.title('Sales vs. Customers Correlation')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.grid()

    # Save the correlation plot as JPG
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\sales_vs_customers_correlation.jpg', format='jpg')
    plt.close()  # Close the figure to save memory

    # Step 8: Output Additional Statistics
    print("\nBasic Statistics for Sales and Customers:")
    print(summary[['Sales', 'Customers']].describe())

    # Step 9: Discuss the Importance of Customer Foot Traffic
    print("\n**Analysis of Sales vs. Customers Correlation:**")
    print("The analysis establishes a strong positive correlation between the number of customers and sales. "
          "This highlights the importance of customer foot traffic in driving revenue. "
          "Understanding this relationship can help in formulating strategies to increase customer visits, "
          "which in turn can lead to higher sales.")

# Step 10: Prepare the Test Data for Predictions
if 'Date' in test_data.columns:
    test_data['Date'] = pd.to_datetime(test_data['Date'], format='%Y-%m-%d')

# (Optional) Further processing or model predictions can be added here
