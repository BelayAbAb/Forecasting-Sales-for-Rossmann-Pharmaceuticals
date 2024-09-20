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

    # Step 7: Outlier Detection using IQR
    def detect_outliers_iqr(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data < lower_bound) | (data > upper_bound)]

    # Detect outliers for Sales and Customers
    sales_outliers = detect_outliers_iqr(daily_summary['Sales'])
    customers_outliers = detect_outliers_iqr(daily_summary['Customers'])

    print("Detected Sales Outliers:")
    print(sales_outliers)
    
    print("\nDetected Customers Outliers:")
    print(customers_outliers)

    # Step 8: Handle Outliers (e.g., Remove or Cap)
    # Here we will remove outliers
    daily_summary = daily_summary[~daily_summary['Sales'].isin(sales_outliers)]
    daily_summary = daily_summary[~daily_summary['Customers'].isin(customers_outliers)]

    # Step 9: Visualize Sales and Customers Trends
    plt.figure(figsize=(14, 7))
    
    # Plot Sales
    plt.subplot(2, 1, 1)
    plt.plot(daily_summary['Date'], daily_summary['Sales'], marker='o', linestyle='-', color='blue', label='Sales')
    plt.title('Daily Sales Over Time (Outliers Removed)')
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
    plt.title('Daily Customers Over Time (Outliers Removed)')
    plt.xlabel('Date')
    plt.ylabel('Total Customers')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()

    # Save Customers Plot
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\daily_customers_plot.jpg', format='jpg')

    plt.tight_layout()
    plt.show()

    # Step 10: Analyze Relationship Between Customers and Sales
    plt.figure(figsize=(8, 6))
    plt.scatter(daily_summary['Customers'], daily_summary['Sales'], alpha=0.5, color='green')
    plt.title('Sales vs. Customers (Outliers Removed)')
    plt.xlabel('Total Customers')
    plt.ylabel('Total Sales')
    plt.grid()
    
    # Save Scatter Plot
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\sales_vs_customers_plot.jpg', format='jpg')
    
    plt.show()

    # Calculate correlation
    correlation = daily_summary['Sales'].corr(daily_summary['Customers'])
    print(f"Correlation between Sales and Customers (Outliers Removed): {correlation:.2f}")

    # Step 11: Output Basic Statistics of Sales and Customers
    print("Basic Statistics of Sales (Target Variable):")
    print(daily_summary['Sales'].describe())
    
    print("\nBasic Statistics of Customers:")
    print(daily_summary['Customers'].describe())

    # Step 12: Emphasize the Importance of Sales and Customers
    print("\nSales is the primary target variable representing daily turnover.")
    print("Customers indicates foot traffic which can influence sales.")
    print("Understanding the relationship between sales and customer visits is crucial for business strategy.")

    # Optional: Save the daily summary data to a CSV for further analysis
    daily_summary.to_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\daily_summary_analysis.csv', index=False)

# Step 13: Prepare the Test Data for Predictions
if 'Date' in test_data.columns:
    test_data['Date'] = pd.to_datetime(test_data['Date'], format='%Y-%m-%d')

# (Optional) Further processing or model predictions can be added here
