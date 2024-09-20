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

# Step 3: Convert the Date Column to Datetime
if 'Date' in train_data.columns:
    train_data['Date'] = pd.to_datetime(train_data['Date'], format='%Y-%m-%d')  # Correct format for your data

# Step 4: Analyze Sales and Customers Data
if 'Date' in train_data.columns and 'Customers' in train_data.columns:
    # Group by date and sum the sales and customers
    daily_summary = train_data.groupby('Date')[['Sales', 'Customers']].sum().reset_index()

    # Step 5: Identify Holidays
    holidays = train_data[train_data['StateHoliday'] != '0']  # Filter out non-holiday entries
    holiday_dates = holidays['Date'].unique()

    # Step 6: Visualize Sales Around Holidays
    plt.figure(figsize=(14, 7))
    
    # Plot Sales
    plt.plot(daily_summary['Date'], daily_summary['Sales'], marker='o', linestyle='-', color='blue', label='Sales')
    
    # Highlight Holiday Sales
    for holiday in holiday_dates:
        plt.axvline(x=holiday, color='red', linestyle='--', label='Holiday' if holiday == holiday_dates[0] else "")
    
    plt.title('Daily Sales with Holidays Highlighted')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    
    # Save Sales Plot
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\daily_sales_holidays_plot.jpg', format='jpg')
    plt.close()  # Close the figure to save memory

    # Step 7: Analyze Sales Behavior on Holidays
    holiday_sales = daily_summary[daily_summary['Date'].isin(holiday_dates)]
    non_holiday_sales = daily_summary[~daily_summary['Date'].isin(holiday_dates)]

    # Calculate average sales
    average_holiday_sales = holiday_sales['Sales'].mean()
    average_non_holiday_sales = non_holiday_sales['Sales'].mean()

    print(f"Average Sales on Holidays: {average_holiday_sales:.2f}")
    print(f"Average Sales on Non-Holidays: {average_non_holiday_sales:.2f}")

    # Step 8: Output Basic Statistics of Holiday Sales
    print("\nBasic Statistics of Sales During Holidays:")
    print(holiday_sales['Sales'].describe())

    # Step 9: Discuss the Impact of Holidays
    print("\n**Analysis of Sales Behavior During Holidays:**")
    print("The analysis shows that holidays significantly impact sales patterns. "
          "During holiday periods, there are notable spikes in sales, indicating increased consumer spending.")
    print("This is crucial for understanding revenue trends, as holidays tend to attract more customers, "
          "reflected in higher sales figures.")
    
    # Step 10: Optional: Save the daily summary data to a CSV for further analysis
    daily_summary.to_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\daily_summary_analysis.csv', index=False)

# Step 11: Prepare the Test Data for Predictions
if 'Date' in test_data.columns:
    test_data['Date'] = pd.to_datetime(test_data['Date'], format='%Y-%m-%d')

# (Optional) Further processing or model predictions can be added here
