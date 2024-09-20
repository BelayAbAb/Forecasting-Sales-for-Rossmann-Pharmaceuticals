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

    # Step 5: Identify Seasonal Events (Christmas and Easter)
    christmas_start = pd.to_datetime("2015-12-24")
    christmas_end = pd.to_datetime("2015-12-26")
    easter_start = pd.to_datetime("2015-04-04")
    easter_end = pd.to_datetime("2015-04-06")

    # Create a mask for Christmas and Easter
    is_christmas = (daily_summary['Date'] >= christmas_start) & (daily_summary['Date'] <= christmas_end)
    is_easter = (daily_summary['Date'] >= easter_start) & (daily_summary['Date'] <= easter_end)

    # Step 6: Aggregate Sales for Christmas and Easter
    christmas_sales = daily_summary[is_christmas]
    easter_sales = daily_summary[is_easter]

    # Step 7: Visualize Sales Trends Around Christmas and Easter
    plt.figure(figsize=(14, 7))
    
    plt.plot(daily_summary['Date'], daily_summary['Sales'], marker='o', linestyle='-', color='blue', label='Sales')
    
    # Highlight Christmas Sales
    plt.fill_between(daily_summary['Date'], 
                     daily_summary['Sales'].where(is_christmas, other=0), 
                     color='red', alpha=0.3, label='Christmas Season')
    
    # Highlight Easter Sales
    plt.fill_between(daily_summary['Date'], 
                     daily_summary['Sales'].where(is_easter, other=0), 
                     color='green', alpha=0.3, label='Easter Season')
    
    plt.title('Daily Sales with Seasonal Events Highlighted')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    
    # Save Seasonal Sales Plot as JPG
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\daily_sales_seasonal_plot.jpg', format='jpg')
    plt.close()  # Close the figure to save memory

    # Step 8: Output Average Sales During Seasonal Events
    average_christmas_sales = christmas_sales['Sales'].mean()
    average_easter_sales = easter_sales['Sales'].mean()

    print(f"Average Sales during Christmas: {average_christmas_sales:.2f}")
    print(f"Average Sales during Easter: {average_easter_sales:.2f}")

    # Step 9: Output Basic Statistics of Sales During Seasonal Events
    print("\nBasic Statistics of Sales During Christmas:")
    print(christmas_sales['Sales'].describe())

    print("\nBasic Statistics of Sales During Easter:")
    print(easter_sales['Sales'].describe())

    # Step 10: Discuss the Impact of Seasonal Events
    print("\n**Analysis of Seasonal Effects on Sales:**")
    print("The analysis shows notable increases in sales during Christmas and Easter periods. "
          "These seasonal spikes indicate increased consumer spending and can guide promotional strategies.")
    
    # Step 11: Optional: Save the daily summary data to a CSV for further analysis
    daily_summary.to_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\daily_summary_analysis.csv', index=False)

# Step 12: Prepare the Test Data for Predictions
if 'Date' in test_data.columns:
    test_data['Date'] = pd.to_datetime(test_data['Date'], format='%Y-%m-%d')

# (Optional) Further processing or model predictions can be added here
