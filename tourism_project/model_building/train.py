# for data manipulation
import pandas as pd
import os
# for preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
# for experimentation tracking
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-prediction")

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_REPO = "ASNaik/tourism-package-prediction"
MODEL_REPO = "ASNaik/tourism-prediction-model"

Xtrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
Xtest = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv").values.ravel()
ytest = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv").values.ravel()

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome"
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation"
]

# Handle class imbalance in the target (ProdTaken)
class_weight = {
    0: 1,
    1: (ytrain == 0).sum() / (ytrain == 1).sum()
}

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Define base model
rf_model = RandomForestClassifier(class_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    "randomforestclassifier__n_estimators": [100, 200, 300],
    "randomforestclassifier__max_depth": [4, 6, 8, None],
    "randomforestclassifier__min_samples_split": [2, 5, 10],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, rf_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring="recall")
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # Log best parameters separately in the main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"],
    })

    # Save the model locally
    model_path = "best_tourism_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Step 1: Check if the model repo already exists
    try:
        api.repo_info(repo_id=MODEL_REPO, repo_type="model")
        print(f"Model repo '{MODEL_REPO}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{MODEL_REPO}' not found. Creating new repo...")
        create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)
        print(f"Model repo '{MODEL_REPO}' created.")

    # Step 2: Upload the trained model
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print("Model uploaded to the Hugging Face Hub.")
