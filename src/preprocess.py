import os
import json
import pandas as pd
import yaml
import opensmile
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))
seed_value = params['preprocess']["random_state"]


def preprocess(params):
    input_path = params['preprocess']["input"]
    output_path = params['preprocess']["output"]
    label2id = {"anger": 0, "surprise": 1, "happiness": 2, "sadness": 3, "neutral": 4, "fear": 5}

    paths = []
    labels = []
    with open(input_path, encoding='utf-8') as ms:
        modified_shemo = json.loads(ms.read())
        for file in modified_shemo:
            path = modified_shemo[file]["path"]
            label = label2id[modified_shemo[file]["emotion"]]
            if label != 5:
                paths.append(f'../{path}')
                labels.append(label)

    data_dict = {'path': paths, 'label': labels}
    df = pd.DataFrame(data_dict)

    features = []
    labels = []

    print("start feature extraction ... ")

    feature_extractor = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        verbose=True, num_workers=None,
        sampling_rate=16000, resample=True)

    for _, row in df.iterrows():
        path = row['path']
        label = row['label']

        df_features = feature_extractor.process_file(f'{path}')
        features.append(df_features.values.squeeze())
        labels.append(label)

    print("feature extraction done! ")

    X = np.array(features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y,
                                                        random_state=seed_value)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    os.makedirs(output_path, exist_ok=True)
    np.save(f"{output_path}X_train.npy", X_train)
    np.save(f"{output_path}X_test.npy", X_test)
    np.save(f"{output_path}y_train.npy", y_train)
    np.save(f"{output_path}y_test.npy", y_test)


if __name__ == "__main__":
    preprocess(params)
