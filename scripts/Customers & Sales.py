import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the Training Data with dtype specifications
train_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\train.csv', 
                          dtype={'StateHoliday': 'str', 'SchoolHoliday': 'int'}, 
                          delimiter=',')  # Use ',' for comma-separated data

# Step 2: Load the Test Data with dtype specifications
test_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\test.csv', 
                         dtype={'StateHoliday': 'str'}, 
                         delimiter=',')

# Step 3: Check Column Names
print("Columns in the Training DataFrame:", train_data.columns)
print("Columns in the Test DataFrame:", test_data.columns)

# Step 4: Inspect the First Few Rows
print("Training Data Sample:")
print(train_data.head())

print("Test Data Sample:")
print(test_data.head())

# Step 5: Convert the Date Column to Datetime if it exists
if 'Date' in train_data.columns:
    train_data['Date'] = pd.to_datetime(train_data['Date'], format='%Y-%m-%d')  # Correct format for your data

# Step 6: Analyze Sales and Customers Data
if 'Date' in train_data.columns and 'Customers' in train_data.columns:
    # Group by date and sum the sales and customers
    daily_summary = train_data.groupby('Date')[['Sales', 'Customers']].sum().reset_index()

    # Step 7: Visualize Sales and Customers Trends
    plt.figure(figsize=(14, 7))
    
    # Plot Sales
    plt.subplot(2, 1, 1)
    plt.plot(daily_summary['Date'], daily_summary['Sales'], marker='o', linestyle='-', color='blue', label='Sales')
    plt.title('Daily Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    
    # Save Sales Plot
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\daily_sales_plot.jpg', format='jpg')

    # Plot Customers
    plt.subplot(2, 1, 2)
    plt.plot(daily_summary['Date'], daily_summary['Customers'], marker='o', linestyle='-', color='orange', label='Customers')
    plt.title('Daily Customers Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Customers')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()

    # Save Customers Plot
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\daily_customers_plot.jpg', format='jpg')

    plt.tight_layout()
    plt.show()

    # Step 8: Analyze Relationship Between Customers and Sales
    plt.figure(figsize=(8, 6))
    plt.scatter(daily_summary['Customers'], daily_summary['Sales'], alpha=0.5, color='green')
    plt.title('Sales vs. Customers')
    plt.xlabel('Total Customers')
    plt.ylabel('Total Sales')
    plt.grid()
    
    # Save Scatter Plot
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\sales_vs_customers_plot.jpg', format='jpg')
    
    plt.show()

    # Calculate correlation
    correlation = daily_summary['Sales'].corr(daily_summary['Customers'])
    print(f"Correlation between Sales and Customers: {correlation:.2f}")

    # Step 9: Output Basic Statistics of Sales and Customers
    print("Basic Statistics of Sales (Target Variable):")
    print(daily_summary['Sales'].describe())
    
    print("\nBasic Statistics of Customers:")
    print(daily_summary['Customers'].describe())

    # Step 10: Emphasize the Importance of Sales and Customers
    print("\nSales is the primary target variable representing daily turnover.")
    print("Customers indicates foot traffic which can influence sales.")
    print("Understanding the relationship between sales and customer visits is crucial for business strategy.")

    # Optional: Save the daily summary data to a CSV for further analysis
    daily_summary.to_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\daily_summary_analysis.csv', index=False)

# Step 11: Prepare the Test Data for Predictions
if 'Date' in test_data.columns:
    test_data['Date'] = pd.to_datetime(test_data['Date'], format='%Y-%m-%d')

# (Optional) Further processing or model predictions can be added here
