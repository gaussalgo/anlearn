import numpy as np

from anlearn import LODA


def test_loda() -> None:
    loda = LODA()

    X = np.random.normal(size=(100, 20))
    loda.fit(X)

    assert loda.score_samples(X).any()
