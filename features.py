import numpy as np, pandas as pd

def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    total = X.drop(columns="Days.Logged.In").sum(axis=1) + 1
    X["total"] = total
    X["read_per_day"] = X["Read.Actions"] / (X["Days.Logged.In"] + 1)
    X["read_share"] = X["Read.Actions"] / total
    X["quiz_share"] = X["Quiz.Actions"] / total
    X["active_share"] = (X["Submission.Actions"] + X["Create.Actions"]) / total
    X["log_read"] = np.log1p(X["Read.Actions"])
    X["has_quiz"] = (X["Quiz.Actions"] > 0).astype(int)
    return X
