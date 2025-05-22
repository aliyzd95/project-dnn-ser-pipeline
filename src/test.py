import yaml
import os
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, confusion_matrix
from tensorflow.keras.models import load_model
import dagshub

dagshub.init(repo_owner='aliyzd95', repo_name='project-dnn-ser-pipeline', mlflow=True)

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/aliyzd95/project-dnn-ser-pipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "aliyzd95"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "5add05d8d42854133eb3f9fe9dcbb57b2360829d"

params = yaml.safe_load(open("params.yaml"))

def plot_confusion_matrix(cm, labels, title, path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def test(params):
    test_path = params["test"]["inputs_path"]
    runs_path = params["test"]["runs_path"]
    models_path = params["test"]["models_path"]
    results_path = params["test"]["results_path"]

    X_test = np.load(f"{test_path}X_test.npy")
    y_test = np.load(f"{test_path}y_test.npy")

    label2id = {
        "anger": 0, "surprise": 1, "happiness": 2,
        "sadness": 3, "neutral": 4, "fear": 5
    }
    id2label = {v: k for k, v in label2id.items()}
    class_labels = [id2label[i] for i in range(len(id2label))]

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment("SER-DNN-experiment")
    os.makedirs("results", exist_ok=True)
    with open(f"{results_path}results.txt", "w") as f:
        f.write("Fold\tRun ID\tTest Loss\tTest Accuracy\tTest UAR\n")

    for fold_no, model_file in enumerate(sorted(os.listdir(models_path))):
        model_path = f"{models_path}{model_file}"

        model = load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        with open(f"{runs_path}run_id_{fold_no + 1}.txt", "r") as f:
            run_id = f.read().strip()

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        test_uar = recall_score(y_test, y_pred, average="macro")
        conf_matrix = confusion_matrix(y_test, y_pred)

        conf_img_path = f"{results_path}conf_matrix_fold_{fold_no+1}.png"

        plot_confusion_matrix(conf_matrix, labels=class_labels,
                              title=f"Confusion Matrix - Fold {fold_no+1}",
                              path=conf_img_path)

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_uar": test_uar,
            })
            mlflow.log_artifact(conf_img_path)

        with open(f"{results_path}results.txt", "a") as f:
            f.write(f"{fold_no + 1}\t{run_id}\t{test_loss}\t{test_acc}\t{test_uar}\n")
            f.write(f"Confusion Matrix:\n{conf_matrix}\n\n")

test(params)
