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

# Step 3: Prepare Data for Visualization
# Combine training and test sets for comparison
train_data['Set'] = 'Training'
test_data['Set'] = 'Test'

# Select relevant columns for analysis
combined_data = pd.concat([train_data[['Promo', 'Set']], test_data[['Promo', 'Set']]], ignore_index=True)

# Step 4: Create Visualizations to Compare Promotional Distributions
plt.figure(figsize=(12, 6))
sns.countplot(data=combined_data, x='Promo', hue='Set', palette='Set2')
plt.title('Distribution of Promotions in Training vs Test Sets')
plt.xlabel('Promo (0 = No Promo, 1 = Promo)')
plt.ylabel('Count')
plt.legend(title='Dataset', loc='upper right')

# Save the promotional distributions plot as JPG
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\promotional_distributions.jpg', format='jpg')
plt.close()

# Summary output for stakeholders
print("\n**Promotional Distribution Analysis:**")
print("The distribution of promotional activities was compared between training and test sets.")
print("Similar distributions indicate a reliable foundation for predictions.")
