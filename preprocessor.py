from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Categorical features
cat_feats = ["gender", "smoking_history"]

# Binary features
bin_feats = ["hypertension", "heart_disease"]

# Numerical features
num_feats = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]


preprocessor = ColumnTransformer(
    [
        ("num", "passthrough", num_feats),
        ("cat", OrdinalEncoder(), cat_feats),
        ("bin", "passthrough", bin_feats),
    ],
    remainder="drop",
    verbose_feature_names_out=True,
)
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
