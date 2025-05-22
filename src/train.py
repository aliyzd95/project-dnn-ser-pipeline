import yaml

params = yaml.safe_load(open("params.yaml"))
seed_value = params["preprocess"]["random_state"]

import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)
import tensorflow as tf

tf.random.set_seed(seed_value)

import optuna
import mlflow
import mlflow.keras
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score
from keras import Input, Model
import pickle
from tensorflow.keras import backend as K
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, SpatialDropout1D, GlobalAveragePooling1D, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import dagshub

dagshub.init(repo_owner='aliyzd95', repo_name='project-dnn-ser-pipeline', mlflow=True)

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/aliyzd95/project-dnn-ser-pipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "aliyzd95"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "5add05d8d42854133eb3f9fe9dcbb57b2360829d"

class UAR(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='uar', **kwargs):
        super(UAR, self).__init__(name=name, **kwargs)
        self.num_classes = 5
        self.conf_matrix = self.add_weight(
            name='conf_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros',
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)

        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, tf.int32)

        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32
        )
        self.conf_matrix.assign_add(current_cm)

    def result(self):
        recall_per_class = tf.linalg.diag_part(self.conf_matrix) / (tf.reduce_sum(self.conf_matrix, axis=1) + 1e-7)
        uar = tf.reduce_mean(recall_per_class)
        return uar

    def reset_states(self):
        self.conf_matrix.assign(tf.zeros_like(self.conf_matrix))


def create_model(N_FEATURES, filters, kernel_size, dropout_rate, learning_rate, activation):
    input_speech = Input((1, N_FEATURES))
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=2, padding='same', activation=activation)(input_speech)
    x = MaxPooling1D(padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(dropout_rate)(x)

    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=2, padding='same', activation=activation)(x)
    x = MaxPooling1D(padding='same')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(dropout_rate)(x)

    x = GlobalAveragePooling1D()(x)
    output = Dense(5, activation='softmax')(x)

    model = Model(input_speech, output)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy', UAR(num_classes=5)])
    return model


def train(params):
    models_path = params["train"]["models_path"]
    runs_path = params["train"]["runs_path"]
    inputs_path = params["train"]["inputs_path"]
    n_trials = params["train"]["n_trials"]
    # n_trials = 10

    X_train = np.load(f"{inputs_path}X_train.npy")
    y_train = np.load(f"{inputs_path}y_train.npy")
    X_test = np.load(f"{inputs_path}X_test.npy")
    y_test = np.load(f"{inputs_path}y_test.npy")

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    signature = mlflow.models.infer_signature(X_train, y_train)

    N_FEATURES = X_train.shape[-1]

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value)

    mlflow.set_experiment("SER-DNN-experiment")
    with mlflow.start_run(run_name="SER-DNN-pipeline"):
        fold_no = 1
        for train_idx, val_idx in kfold.split(X_train, y_train):
            X_tune_train, X_tune_val = X_train[train_idx], X_train[val_idx]
            y_tune_train, y_tune_val = y_train[train_idx], y_train[val_idx]

            def objective(trial):
                filters = trial.suggest_categorical('filters', [64, 128, 256])
                kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
                dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
                learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
                activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'swish'])

                model = create_model(N_FEATURES, filters, kernel_size, dropout_rate, learning_rate, activation)

                model.fit(
                    X_tune_train, y_tune_train,
                    validation_data=(X_tune_val, y_tune_val),
                    epochs=20,
                    batch_size=32,
                    verbose=0,
                    callbacks=[EarlyStopping('val_uar', patience=5, mode='max', verbose=0)]
                )

                y_val_pred = np.argmax(model.predict(X_tune_val, verbose=0), axis=1)
                uar = recall_score(y_tune_val, y_val_pred, average='macro')

                return uar

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_trial.params

            model = create_model(N_FEATURES, **best_params)
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=32,
                verbose=2,
                callbacks=[EarlyStopping('val_uar', patience=10, restore_best_weights=True, mode='max', verbose=2)]
            )

            with mlflow.start_run(run_name=f"Fold_{fold_no}", nested=True) as run:
                run_id = run.info.run_id

                os.makedirs(runs_path, exist_ok=True)
                with open(f"{runs_path}run_id_{fold_no}.txt", "w") as f:
                    f.write(run_id)

                mlflow.log_params(best_params)
                mlflow.keras.log_model(model, f"model_{fold_no}", signature=signature)

                os.makedirs(models_path, exist_ok=True)
                model.save(f"{models_path}model_{fold_no}.keras")
                fold_no += 1


if __name__ == "__main__":
    train(params)
