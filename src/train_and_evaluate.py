import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from get_data import read_params
import argparse
import joblib
import json

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]

    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    C = config["estimators"]["SVM"]["C"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)
    svm = SVC(C=C, random_state=random_state)
    svm.fit(train_x, train_y.values.ravel())

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(svm, model_path)
    predicted_note = svm.predict(test_x)
    fscore= f1_score(test_y, predicted_note)
    print("C={} and score={}".format(C, fscore))
    scores_files = config["reports"]["scores"]
    params_files = config["reports"]["params"]

    with open(params_files, "w") as f:
        params = {
            "C": C
        }
        json.dump(params, f, indent=4)

    with open(scores_files, "w") as f:
        scores = {
            "fscore": fscore
        }
        json.dump(scores, f, indent=4)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args= args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)