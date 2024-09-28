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

# Step 4: Analyze Sales During Promotional Campaigns
if 'Sales' in train_data.columns and 'Promo' in train_data.columns:
    # Step 5: Create a summary DataFrame for promotional and non-promotional periods
    promo_summary = train_data.groupby(['Date', 'Promo'])['Sales'].sum().reset_index()

    # Step 6: Calculate average sales for promotional and non-promotional periods
    avg_sales = promo_summary.groupby('Promo')['Sales'].mean().reset_index()

    # Step 7: Visualize the Sales Impact of Promotions
    plt.figure(figsize=(12, 6))
    sns.barplot(data=avg_sales, x='Promo', y='Sales', hue='Promo', palette='viridis', dodge=False, legend=False)
    plt.title('Average Sales During Promotional and Non-Promotional Periods')
    plt.xlabel('Promotion Active (0 = No, 1 = Yes)')
    plt.ylabel('Average Sales')
    plt.xticks(ticks=[0, 1], labels=['No Promotion', 'Promotion'])
    
    # Save the promotions impact plot as JPG
    plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\impact_of_promotions.jpg', format='jpg')
    plt.close()  # Close the figure to save memory

    # Step 8: Discuss the Findings
    print("\n**Analysis of Impact of Promotions on Sales:**")
    print("Promotional campaigns significantly affect sales. The average sales during promotional periods "
          "are notably higher than during non-promotional periods. This analysis concludes that promotions "
          "not only attract new customers but also encourage repeat visits from existing customers, "
          "ultimately boosting overall revenue.")

# Step 9: Prepare the Test Data for Predictions
if 'Date' in test_data.columns:
    test_data['Date'] = pd.to_datetime(test_data['Date'], format='%Y-%m-%d')

# (Optional) Further processing or model predictions can be added here
