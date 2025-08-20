import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load and preprocess the dataset
# Load the dataset
df = pd.read_csv('/home/danny/Downloads/dataset.csv')


# Function to preprocess the dataframe and handle percentage values
def preprocess_dataframe(df):
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    # Loop through each column to detect and convert percentage values
    for column in processed_df.columns:
        # Check if the column contains string values
        if processed_df[column].dtype == 'object':
            # Check if column has data
            if not processed_df[column].dropna().empty:
                # Sample the first non-null value
                sample = processed_df[column].dropna().iloc[0]

                # If it's a string and contains '%', convert it
                if isinstance(sample, str) and '%' in sample:
                    processed_df[column] = processed_df[column].apply(
                        lambda x: float(x.replace('%', '').strip()) / 100 if isinstance(x, str) and '%' in x else x
                    )

    return processed_df


# Apply preprocessing
df = preprocess_dataframe(df)

# 2. Feature Selection (choose most relevant for risk classification)
risk_features = [
    # Market metrics
    'Beta', 'Monthly Volatility', 'Weekly Volatility', 'Sharpe Ratio',

    # Valuation metrics
    'Trailing PE', 'Forward PE', 'P/B Ratio',

    # Financial health
    'D/E Ratio', 'Current Ratio', 'Quick Ratio', 'Cash To Debt',

    # Growth and profitability
    'EPS Growth Next 5yr', 'EPS Growth Past 5yr', 'ROE', 'Profit Margin',

    # Dividend related
    'Div Yield', 'Consecutive Yrs Div Increase', 'Dividend Rank',

    # Size and stability
    'Market Cap', 'Piotroski F', 'Altman Z',

    # Categorical features
    'Sector', 'Asset Class', 'Cap Size'
]

# Filter features that exist in the dataset
available_features = [f for f in risk_features if f in df.columns]
X = df[available_features]


# 3. Create Risk Labels (Target Variable)
def create_risk_score(row):
    score = 0

    # Beta contribution (higher beta = higher risk)
    if pd.notna(row.get('Beta')):
        if row['Beta'] < 0.8:
            score += 0
        elif row['Beta'] < 1.2:
            score += 5
        else:
            score += 10

    # Volatility contribution
    vol_col = 'Monthly Volatility'
    if vol_col in row and pd.notna(row[vol_col]):
        try:
            vol = float(row[vol_col])
            score += min(int(vol * 100), 10)  # Cap at 10 points
        except (ValueError, TypeError):
            pass  # Skip if conversion fails

    # Market cap (smaller = higher risk)
    if pd.notna(row.get('Market Cap')):
        if row['Market Cap'] > 10e9:  # Large cap
            score += 0
        elif row['Market Cap'] > 2e9:  # Mid cap
            score += 5
        else:  # Small cap
            score += 10

    # Dividend stability (more stable = lower risk)
    div_col = 'Consecutive Yrs Div Increase'
    if div_col in row and pd.notna(row[div_col]):
        try:
            cons_years = float(row[div_col])
            score -= min(int(cons_years), 10)  # Reduce score up to 10 points
        except (ValueError, TypeError):
            pass  # Skip if conversion fails

    # Financial health (higher = lower risk)
    if pd.notna(row.get('Altman Z')):
        if row['Altman Z'] > 3:
            score -= 5
        elif row['Altman Z'] < 1.8:
            score += 5

    return score


# Apply function to create risk scores
df['Risk_Score'] = df.apply(create_risk_score, axis=1)

# Create risk categories
df['Risk_Category'] = pd.cut(df['Risk_Score'],
                             bins=[-float('inf'), 0, 15, float('inf')],
                             labels=['Low', 'Medium', 'High'])

# Our target variable
y = df['Risk_Category']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 5. Preprocessing and Feature Engineering
# Identify numerical and categorical columns
categorical_features = [f for f in available_features if f in ['Sector', 'Asset Class', 'Cap Size']]
numerical_features = [f for f in available_features if f not in categorical_features]

# Create preprocessing steps for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create preprocessing steps for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 6. SVM Model Building
# Create an SVM pipeline
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
])

# Train the model
svm_pipeline.fit(X_train, y_train)

# 7. Model Evaluation
# Make predictions
y_pred = svm_pipeline.predict(X_test)

# Print evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Hyperparameter Tuning
# Define parameter grid
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': [0.01, 0.1, 1, 'scale'],
    'classifier__kernel': ['rbf', 'poly', 'sigmoid']
}

# Create grid search
grid_search = GridSearchCV(
    svm_pipeline,
    param_grid,
    cv=5,
    scoring='f1_weighted',
    verbose=1,
    n_jobs=-1  # Use all available cores
)

# Fit grid search
print("\nPerforming hyperparameter tuning. This may take some time...")
grid_search.fit(X_train, y_train)

# Print best parameters and score
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("\nBest Model Classification Report:")
print(classification_report(y_test, y_pred_best))

# 9. Save the Model
joblib.dump(best_model, 'stock_risk_classifier_svm.pkl')
print("\nModel saved as 'stock_risk_classifier_svm.pkl'")


# 10. Function for Classifying New Stocks
def classify_stock_risk(stock_data):
    """
    Classify a stock's risk profile using the trained SVM model

    Parameters:
    stock_data (DataFrame): DataFrame containing stock features

    Returns:
    str: Risk category ('Low', 'Medium', 'High')
    float: Probability of the predicted class
    dict: Class probabilities
    """
    # Ensure data has the right format
    stock_features = stock_data[available_features]

    # Preprocess the data
    stock_features = preprocess_dataframe(stock_features)

    # Load the model
    model = joblib.load('stock_risk_classifier_svm.pkl')

    # Make prediction
    risk_category = model.predict(stock_features)[0]

    # Get probabilities
    probabilities = model.predict_proba(stock_features)[0]

    # Get the probability of the predicted class
    class_indices = {label: idx for idx, label in enumerate(model.classes_)}
    predicted_prob = probabilities[class_indices[risk_category]]

    # Create probability dictionary
    prob_dict = {label: probabilities[idx] for label, idx in class_indices.items()}

    return risk_category, predicted_prob, prob_dict


# Example of how to use the classification function
print("\nExample of classification function usage:")
print("(To use this, you would provide a DataFrame with the required features)")
print("result = classify_stock_risk(new_stock_data)")