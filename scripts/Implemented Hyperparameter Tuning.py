import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the data with low_memory=False to suppress DtypeWarning
train_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\train.csv', delimiter=',', low_memory=False)
store_data = pd.read_csv(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\store.csv', delimiter=',', low_memory=False)

# Merge the train and store data
dataset = train_data.merge(store_data, on='Store')

# Convert 'Date' to datetime using the correct format
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%Y-%m-%d')

# Extract features and target variable
X = dataset.drop(columns=['Sales', 'Customers'])
y = dataset['Sales']

# Define numerical and categorical features
numerical_features = ['Open', 'Promo', 'SchoolHoliday', 'CompetitionDistance']
categorical_features = ['StateHoliday', 'StoreType', 'Assortment', 'DayOfWeek']

# Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Impute numerical features
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute categorical features
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create the complete pipeline with Random Forest Regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5],
    'model__max_features': ['sqrt', 'log2']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Output best parameters
print("Best Parameters:", grid_search.best_params_)

# Extract the best model
best_model = grid_search.best_estimator_.named_steps['model']

# Get feature names from the preprocessor
num_features = pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['imputer'].get_feature_names_out(numerical_features)
cat_features = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['imputer'].get_feature_names_out(categorical_features)

# Combine numerical and categorical feature names
feature_names = np.concatenate([num_features, cat_features])

# Get feature importances
importances = best_model.feature_importances_

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.barh(feature_names, importances, color='lightblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importances from Random Forest Regressor')

# Save the feature importance plot as JPG
plt.savefig(r'C:\Users\User\Desktop\10Acadamy\Week 4\Week-4\Data\feature_importances.jpg', format='jpg')
plt.close()
