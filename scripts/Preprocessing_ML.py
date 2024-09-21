import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the datasets
train_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\train.csv', 
                          delimiter=',', 
                          dtype={'StateHoliday': 'str', 'SchoolHoliday': 'int'})

test_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\test.csv', 
                         delimiter=',', 
                         dtype={'StateHoliday': 'str'})

store_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\store.csv', delimiter=',')

# Feature extraction function
def parse_dates(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=False)  # Set dayfirst=False
    df['Weekday'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = (df['Weekday'] >= 5).astype(int)
    return df

# Parse dates for both train and test data
train_data = parse_dates(train_data)
test_data = parse_dates(test_data)

# Merge with store data
train_data = train_data.merge(store_data, on='Store', how='left')
test_data = test_data.merge(store_data, on='Store', how='left')

# One-Hot Encoding
categorical_features = ['StoreType', 'Assortment', 'StateHoliday']

encoder = OneHotEncoder(sparse=False, drop='first')
train_encoded = encoder.fit_transform(train_data[categorical_features])
test_encoded = encoder.transform(test_data[categorical_features])

# Convert to DataFrame
train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_features))
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_features))

# Combine encoded features with original data
train_data = pd.concat([train_data.drop(columns=categorical_features), train_encoded_df], axis=1)
test_data = pd.concat([test_data.drop(columns=categorical_features), test_encoded_df], axis=1)

# Define numerical features to scale
# Only include features that are present in the train and test sets
numerical_features = ['Open', 'Promo', 'SchoolHoliday', 'Weekday', 'IsWeekend']

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])

# For test data, only scale existing features
test_data[numerical_features] = scaler.transform(test_data[numerical_features])  # Scale the test data

# Display the processed data
print("Processed Training Data:")
print(train_data.head())

print("\nProcessed Test Data:")
print(test_data.head())
