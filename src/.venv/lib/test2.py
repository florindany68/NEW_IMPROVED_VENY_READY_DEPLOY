
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. Read CSV
df = pd.read_csv("/home/danny/Downloads/dataset.csv")

# 2. Filter to new/inexperienced investors if there's an "experience" column
# Uncomment or adapt if your dataset includes user-specific data
# df = df[df["years_investing"] == 0]

# 3. Classify risk (sample rule using volatility)


# 4. Select relevant columns (example columns, adapt to your data)
columns_to_keep = [

    "beta",
    "market_cap",
    "historical_return",
    "risk_category"
]

# Drop rows with missing data in these columns
df_svm = df[columns_to_keep].dropna()

# 5. Split into features (X) and target (y)
X = df_svm.drop("risk_category", axis=1)
y = df_svm["risk_category"]

# 6. Label-encode target if needed for SVM classification
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Now you have X and y_encoded for your SVM model
