hyperparameters = {
    "colsample_bylevel": 0.7000000000000001,
    "colsample_bytree": 0.6000000000000001,
    "gamma": 5.0,
    "learning_rate": 0.12,
    "max_depth": 0,
    "n_estimators": 88,
    "scale_pos_weight": 6.0,
    "subsample": 0.6000000000000001,
}
# take a look in nb how and this threshold was decided
threshold = 0.35

feats = [
    "num__age",
    "num__bmi",
    "num__HbA1c_level",
    "num__blood_glucose_level",
    "cat__gender",
    "cat__smoking_history",
    "bin__hypertension",
    "bin__heart_disease",
]
target = "diabetes"
