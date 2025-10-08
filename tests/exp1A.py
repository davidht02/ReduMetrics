

# A1 — Identidad e isometrías (sin reducir dimensión)
import numpy as np
import pandas as pd

# 1) Datos base
SEED = 1234
rng = np.random.default_rng(SEED)
m, n, C = 1200, 10, 4   # puedes ajustar
from sklearn.datasets import make_blobs
X_high, labels = make_blobs(n_samples=m, n_features=n, centers=C,
                            cluster_std=1.0, random_state=SEED)
# estandarización ligera (opcional, útil con distancia euclídea)
X_high = (X_high - X_high.mean(axis=0, keepdims=True)) / X_high.std(axis=0, keepdims=True)

# 2) Isometrías “de andar por casa”: identidad, traslación, escala, rotación en (x0,x1)
# identidad
X_low_id = X_high.copy()

# traslación: suma un vector constante (no cambia distancias)
b = np.full((1, n), 0.5)
X_low_trans = X_high + b

# escala uniforme: multiplica por escalar > 0 (preserva orden y vecindades)
s = 2.0
X_low_scale = s * X_high

# rotación en el plano (x0,x1): matriz identidad con bloque 2x2 de rotación
theta = np.deg2rad(30.0)  # 30 grados
R2 = np.array([[np.cos(theta), -np.sin(theta)],
               [np.sin(theta),  np.cos(theta)]], dtype=float)
Q = np.eye(n)
Q[:2, :2] = R2
X_low_rot = X_high @ Q

# 3) Evalúa las métricas en cada variante
from ReduMetrics.metrics.ulse import ulse_score_sklearn
from ReduMetrics.metrics.rta import rta_score
from ReduMetrics.metrics.spearman import spearman_correlation
from ReduMetrics.metrics.k_ncp import kncp_score
from ReduMetrics.metrics.cdc import cdc_score

K_DEFAULT = 10
T_TRIPLETS = 10_000
P_PAIRS = 10_000

def all_metrics(Xh, Xl, y):
    return {
        "ULSE": float(ulse_score_sklearn(Xh, Xl, k=K_DEFAULT)),
        "RTA": float(rta_score(Xh, Xl, T=T_TRIPLETS, random_state=SEED)),
        "Spearman": float(spearman_correlation(Xh, Xl, P=P_PAIRS, random_state=SEED)),
        "k-NCP": float(kncp_score(Xh, Xl, y)),
        "CDC": float(cdc_score(Xh, Xl, y)),
    }

res_A1 = pd.DataFrame.from_dict({
    "Identidad":     all_metrics(X_high, X_low_id,   labels),
    "Traslación":    all_metrics(X_high, X_low_trans,labels),
    "Escala":        all_metrics(X_high, X_low_scale,labels),
    "Rotación 2D":   all_metrics(X_high, X_low_rot,  labels),
}, orient="index")

print(res_A1)
