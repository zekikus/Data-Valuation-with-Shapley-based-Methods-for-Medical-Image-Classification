import pickle
import medmnist
import numpy as np
import pandas as pd
from medmnist import INFO

def read_pickle(path = None):
    data = []
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data 

def select_top_k_samples(data_flag = None, k = None, pickle_path = None):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    dataset = DataClass(split = 'train')

    shap_values = read_pickle(pickle_path)
    labels = dataset.labels
    indexes = np.array(range(len(labels)))
    indexes = np.expand_dims(indexes, axis = 1)
    shap_data = np.concatenate((shap_values.reshape(len(shap_values), 1), labels.reshape(len(labels), 1), indexes), axis = 1)

    nbr_classes = len(info['label'])
    selected_indexes = list()
    for i in range(nbr_classes):
        filtered = shap_data[np.where(shap_data[:, 1] == i)]
        selected = filtered[filtered[:, 0].argsort()][-(k // nbr_classes):]
        #-np.sort(-filtered, axis = 0)[:round(k // nbr_classes)]
        selected_indexes.extend(selected[:, 2])

    return [int(i) for i in selected_indexes]

def select_top_k_samples_random(data_flag = None, k = None):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    dataset = DataClass(split = 'train')

    labels = dataset.labels
    indexes = np.array(range(len(labels)))
    indexes = np.expand_dims(indexes, axis = 1)
    data = np.concatenate((labels.reshape(len(labels), 1), indexes), axis = 1)

    nbr_classes = len(info['label'])
    selected_indexes = list()
    for i in range(nbr_classes):
        filtered = data[np.where(data[:, 0] == i)]
        np.random.seed(42)
        selected = np.random.choice(filtered[:,1], min(k // nbr_classes, len(filtered)), replace=False)
        selected_indexes.extend(selected)

    return [int(i) for i in selected_indexes]

def select_top_k_samples_deep_explainer(data_flag = None, k = None, file_path = None):
    df = pd.read_excel(file_path)
    info = INFO[data_flag]
    shap_data = df.to_numpy()
    del df

    nbr_classes = len(info['label'])
    selected_indexes = list()
    for i in range(nbr_classes):
        filtered = shap_data[np.where(shap_data[:, 1] == i)]
        selected = filtered[filtered[:, 2].argsort()][-(k // nbr_classes):]
        selected_indexes.extend(selected[:, 0])

    return [int(i) for i in selected_indexes]

def select_top_k_samples_forgetting(data_flag = None, k = None, file_path = None):
    shap_data = None
    with open(f"{file_path}", "rb") as f:
        shap_data = pickle.load(f)
    info = INFO[data_flag]

    nbr_classes = len(info['label'])
    selected_indexes = list()
    for i in range(nbr_classes):
        filtered = shap_data[np.where(shap_data[:, 1] == i)]
        selected = filtered[filtered[:, 2].argsort()][-(k // nbr_classes):]
        selected_indexes.extend(selected[:, 0])

    return [int(i) for i in selected_indexes]

def get_selected_indexes(data_flag = None, k = None, path = None, selection_method = None):
    if selection_method in ['tmc', 'tmc_v2', 'tmc_v3']:
        print("Selection Method:", selection_method)
        return select_top_k_samples(data_flag = data_flag, k = k, pickle_path = f"shapley_files/{path}")
    elif selection_method == 'deep_explainer':
        print("Selection Method:", selection_method)
        return select_top_k_samples_deep_explainer(data_flag=data_flag, k = k, file_path=f"shapley_files/{path}")
    elif selection_method == 'random':
        print("Selection Method: Random")
        return select_top_k_samples_random(data_flag = data_flag, k = k)
    elif selection_method == 'forgetting':
        print("Selection Method: Forgetting")
        return select_top_k_samples_forgetting(data_flag = data_flag, k = k, file_path = f"shapley_files/{path}")