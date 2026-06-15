import numpy as np
import math
import pandas as pd
import warnings

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import pairwise_distances, rand_score
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
np.random.seed(42)




# -------------------- COMMON --------------------




def assign_clusters(X, C):
    return np.argmin(pairwise_distances(X, C), axis=1)

def quantization_error(X, labels, C):
    n, d = X.shape
    return np.sum(np.linalg.norm(X - C[labels], axis=1)**2) / (n * d)

def intra_cluster_distance(X, labels, C):
    n, d = X.shape
    return np.mean(np.linalg.norm(X - C[labels], axis=1)) / np.sqrt(d)

def inter_cluster_distance(C):
    k = len(C)
    D = pairwise_distances(C)
    return (2 / (k * (k - 1))) * np.sum(D[np.triu_indices(k, 1)])

def distance_index(X, labels, C):
    qe = quantization_error(X, labels, C)
    icd = intra_cluster_distance(X, labels, C)
    inter = inter_cluster_distance(C)
    return (qe * icd) / inter


# -------------------- ACO --------------------



def aco_initialize(X, k, ants=30):
    best_C, best_val = None, np.inf
    for _ in range(ants):
        C = X[np.random.choice(len(X), k, replace=False)]
        val = quantization_error(X, assign_clusters(X, C), C)
        if val < best_val:
            best_val, best_C = val, C.copy()
    return best_C




# -------------------- MBF --------------------




def mbf_update(X, C, alpha=0.4):
    labels = assign_clusters(X, C)
    for j in range(len(C)):
        pts = X[labels == j]
        if len(pts) > 0:
            C[j] += alpha * (pts.mean(axis=0) - C[j])
    return C



# -------------------- LEVY --------------------



def levy_flight(beta, dim):
    sigma = (
        math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
        (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    return np.random.randn(dim) * sigma / (np.abs(np.random.randn(dim)) ** (1 / beta))




# -------------------- MUTATION --------------------



def mutation(X, C, labels, eta=0.7, scale=0.12, beta=1.5):
    new_C = C.copy()
    for j in range(len(C)):
        pts = X[labels == j]
        if len(pts) > 0:
            mu = pts.mean(axis=0)
            new_C[j] = C[j] + eta * (mu - C[j]) + scale * levy_flight(beta, C.shape[1])
    return new_C



# -------------------- ILS --------------------



def ils(X, C, steps=8, step_size=0.02):
    best_C = C.copy()
    best_qe = quantization_error(X, assign_clusters(X, best_C), best_C)

    for _ in range(steps):
        cand = best_C + np.random.uniform(-step_size, step_size, best_C.shape)
        qe = quantization_error(X, assign_clusters(X, cand), cand)
        if qe < best_qe:
            best_C, best_qe = cand, qe

    return best_C



# -------------------- MAIN ALGORITHM --------------------



def adaptive_mutaswarmclus(X, k, iters=30):
    C = aco_initialize(X, k)

    for _ in range(iters):
        C = mbf_update(X, C)
        labels = assign_clusters(X, C)

        C_mut = mutation(X, C, labels)

        if quantization_error(X, assign_clusters(X, C_mut), C_mut) < \
           quantization_error(X, labels, C):
            C = C_mut

        C = ils(X, C)

    labels = assign_clusters(X, C)

    return {
        "QE": quantization_error(X, labels, C),
        "ICD": intra_cluster_distance(X, labels, C),
        "INTER": inter_cluster_distance(C),
        "DI": distance_index(X, labels, C),
        "labels": labels
    }



# -------------------- MBF ALGORITHM --------------------


def mbf_clustering(X, k, iters=30):
    # Random initialization
    C = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(iters):
        labels = assign_clusters(X, C)
        for j in range(k):
            pts = X[labels == j]
            if len(pts) > 0:
                C[j] = C[j] + 0.4 * (pts.mean(axis=0) - C[j])

    labels = assign_clusters(X, C)

    return {
        "QE": quantization_error(X, labels, C),
        "ICD": intra_cluster_distance(X, labels, C),
        "INTER": inter_cluster_distance(C),
        "DI": distance_index(X, labels, C),
        "labels": labels
    }




# -------------------- K-MEANS ALGORITHM --------------------


def kmeans_clustering(X, k):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X)
    C = model.cluster_centers_

    return {
        "QE": quantization_error(X, labels, C),
        "ICD": intra_cluster_distance(X, labels, C),
        "INTER": inter_cluster_distance(C),
        "DI": distance_index(X, labels, C),
        "labels": labels
    }



# -------------------- DATA LOADER --------------------


# FOR 5 datasets TEST 
# -------------------

# def load_data():
#     def openml(id, k):
#         data = fetch_openml(data_id=id, as_frame=True)
#         X = data.data.apply(pd.to_numeric, errors="coerce")
#         X = SimpleImputer(strategy="mean").fit_transform(X)
#         X = MinMaxScaler().fit_transform(X)
#         y = LabelEncoder().fit_transform(data.target.astype(str))
#         return X, y, k

#     return {
#         "Iris": (*load_iris(return_X_y=True), 3),
#         "Wine": (*load_wine(return_X_y=True), 3),
#         "Breast Cancer": (*load_breast_cancer(return_X_y=True), 2),
#         "Seeds": openml(1499, 3),
#     }



# FOR 10 datasets TEST 
# -------------------


def load_data():
    def openml(id, k):
        data = fetch_openml(data_id=id, as_frame=True)
        X = data.data.apply(pd.to_numeric, errors="coerce")
        X = SimpleImputer(strategy="mean").fit_transform(X)
        X = MinMaxScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(data.target.astype(str))
        return X, y, k

    return {
        "Iris": (*load_iris(return_X_y=True), 3),
        "Wine": (*load_wine(return_X_y=True), 3),
        "Breast Cancer": (*load_breast_cancer(return_X_y=True), 2),
        "Seeds": openml(1499, 3),
        "Thyroid": openml(40705, 3),
        "Ionosphere": openml(59, 2),
        "Image Segmentation": openml(40984, 7),
        "Vote": openml(1240, 2),
        "Dermatology": openml(35, 6),
        "Vehicle": openml(54, 4),
    }



# -------------------- RUN --------------------



if __name__ == "__main__":
    datasets = load_data()

    print("\nFinal Results (All Metrics)")
    print("-----------------------------")

    for name, (X, y, k) in datasets.items():
        X = MinMaxScaler().fit_transform(X)

        print(f"\n{name}")
        print("--------------------")

        # 🔹 Adaptive
        res_adaptive = adaptive_mutaswarmclus(X, k)
        rc_adaptive = 1.0 - rand_score(y, res_adaptive["labels"])

        print("Adaptive-MutaSwarmClus")
        print(f"QE   : {res_adaptive['QE']:.4f}")
        print(f"ICD  : {res_adaptive['ICD']:.4f}")
        print(f"INTER: {res_adaptive['INTER']:.4f}")
        print(f"DI   : {res_adaptive['DI']:.4f}")
        print(f"RC   : {rc_adaptive:.4f}")


        #FOR 


        # 🔹 MBF
        res_mbf = mbf_clustering(X, k)
        rc_mbf = 1.0 - rand_score(y, res_mbf["labels"])

        print("\nMBF")
        print(f"QE   : {res_mbf['QE']:.4f}")
        print(f"ICD  : {res_mbf['ICD']:.4f}")
        print(f"INTER: {res_mbf['INTER']:.4f}")
        print(f"DI   : {res_mbf['DI']:.4f}")
        print(f"RC   : {rc_mbf:.4f}")

        # 🔹 K-Means
        res_kmeans = kmeans_clustering(X, k)
        rc_kmeans = 1.0 - rand_score(y, res_kmeans["labels"])

        print("\nK-Means")
        print(f"QE   : {res_kmeans['QE']:.4f}")
        print(f"ICD  : {res_kmeans['ICD']:.4f}")
        print(f"INTER: {res_kmeans['INTER']:.4f}")
        print(f"DI   : {res_kmeans['DI']:.4f}")
        print(f"RC   : {rc_kmeans:.4f}")
