import numpy as np
from scipy.sparse import rand

from anlearn.loda import LODA


def test_loda_fit_matrix() -> None:
    loda = LODA(n_estimators=100, bins=10, random_state=42)

    X = np.array([[0, 0], [0.1, -0.2], [0.3, 0.2], [0.2, 0.2], [-5, -5], [0.6, 0.7]])

    loda.fit(X)

    loda.predict(X)

    loda.score_samples(X)

    loda.score_features(X)


def test_loda_fit_sparse_matrix() -> None:
    loda = LODA(n_estimators=100, bins=10, random_state=42)

    matrix = rand(500, 30, density=0.25, format="csr", random_state=42)

    loda.fit(matrix)

    loda.predict(matrix)

    loda.score_samples(matrix)

    loda.score_features(matrix)
