import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def read_from_csv(filename, label_name):
    df = pd.read_csv(filename)
    # print(df)

    labels = torch.tensor(df[label_name].to_numpy(), dtype=torch.float32)
    features = torch.tensor(df.drop(columns=label_name).values, dtype=torch.float32)
    # print(features)

    # print(labels.shape, features.shape)
    return labels,features
    
# read_from_csv("trim32_without_rownames.csv")


def normalize_meat_data(features, y):
    #would be nice to remove some outliers
    # print(features.shape)
    # test_tensor = torch.tensor([[3,1],[9,2]]).float()
    avg = torch.mean(features, dim=0)
    stddev = torch.std(features, dim=0)
    normalized_features = (features - avg)/stddev

    log_fat = torch.log(y)
    mean = torch.mean(log_fat)
    std = torch.std(log_fat, unbiased=False)
    normalized_fat = (log_fat - mean) / std
    return normalized_features, normalized_fat

def applyPCA(X_train, X_test):
    pca = PCA(n_components=30)
    # print("shapes", X_train.shape, X_test.shape)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Step 5: Scale and center PCs
    scaler_pcs = StandardScaler()
    X_train_pcs_scaled = scaler_pcs.fit_transform(X_train_pca)
    X_test_pcs_scaled = scaler_pcs.transform(X_test_pca)

    return X_train_pcs_scaled, X_test_pcs_scaled