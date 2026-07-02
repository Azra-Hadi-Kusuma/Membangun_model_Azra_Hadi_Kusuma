import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='titanic_preprocessed.csv',
                        help='Path ke dataset preprocessed')
    args = parser.parse_args()

    # 1. Load data
    df = pd.read_csv(args.train_data)
    X = df.drop('survived', axis=1)
    y = df['survived']

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Setup MLflow
    mlflow.set_experiment("Titanic_Experiment")

    with mlflow.start_run(run_name="RandomForest_Basic") as run:
        # 3a. Aktifkan autolog (mencatat parameter, metrik, dan model)
        mlflow.sklearn.autolog()

        # 3b. Training model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 3c. Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(f"Run ID: {run.info.run_id}")

        # 3d. Log model secara eksplisit (agar muncul di Artifacts)
        #     Gunakan parameter 'name' (bukan 'artifact_path') sesuai warning MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="Titanic_RandomForest"
        )

        print("Model dan artifacts berhasil di-log. Cek MLflow UI.")

if __name__ == "__main__":
    main()