import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=False)
    df['Weekday'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = (df['Weekday'] >= 5).astype(int)
    return df

# Parse dates for both train and test data
train_data = parse_dates(train_data)
test_data = parse_dates(test_data)

# Merge with store data
train_data = train_data.merge(store_data, on='Store', how='left')
test_data = test_data.merge(store_data, on='Store', how='left')

# Define feature sets
features = ['Open', 'Promo', 'SchoolHoliday', 'Weekday', 'IsWeekend', 'StoreType', 'Assortment', 'StateHoliday']
target = 'Sales'

# Split the training data into features and target
X = train_data[features]
y = train_data[target]

# Create a Column Transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values for numerical features
            ('scaler', StandardScaler())
        ]), ['Open', 'Promo', 'SchoolHoliday', 'Weekday', 'IsWeekend']),
        ('cat', OneHotEncoder(drop='first'), ['StoreType', 'Assortment', 'StateHoliday'])
    ]
)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit the model
pipeline.fit(X, y)

# Prepare the test data
X_test = test_data[features]

# Make predictions on the test set
predictions = pipeline.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y, pipeline.predict(X))
mse = mean_squared_error(y, pipeline.predict(X))

# Prepare the submission DataFrame
submission = pd.DataFrame({'Id': test_data['Id'], 'Sales': predictions})

# Save the submission DataFrame to a CSV file
submission.to_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\submission.csv', index=False)

# Feature importances
model = pipeline.named_steps['model']
feature_names = (pipeline.named_steps['preprocessor']
                 .transformers_[0][1]
                 .named_steps['scaler']
                 .get_feature_names_out(input_features=['Open', 'Promo', 'SchoolHoliday', 'Weekday', 'IsWeekend']).tolist()
                 + list(pipeline.named_steps['preprocessor']
                         .transformers_[1][1]
                         .get_feature_names_out()))

importances = model.feature_importances_

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.barh(feature_names, importances, color='lightblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importances from Random Forest Regressor')

# Save the feature importance plot as JPG
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\feature_importances.jpg', format='jpg')
plt.close()

# Print model statistics and explain choice of MAE
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Add explanation for MAE choice
print("\n**Model Evaluation:**")
print("Selected Mean Absolute Error (MAE) as the loss function due to its interpretability and robustness against outliers, making it suitable for assessing forecasting accuracy in this context.")

# Plot MAE and MSE
plt.figure(figsize=(8, 4))
plt.bar(['MAE', 'MSE'], [mae, mse], color=['blue', 'orange'])
plt.ylabel('Error Value')
plt.title('Model Performance Metrics')

# Save the performance metrics plot as JPG
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\performance_metrics.jpg', format='jpg')
plt.close()

print("Model training and prediction completed, and statistics saved.")
