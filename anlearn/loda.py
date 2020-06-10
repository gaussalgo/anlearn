from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from ._typing import ArrayLike


class HistPdf:
    def __init__(self, bins: Union[int, str] = "auto", return_min: bool = True) -> None:
        self.bins = bins
        self.return_min = return_min

    def fit(self, X: np.ndarray) -> "HistPdf":
        self.histogram = np.histogram(X, bins=self.bins)
        hist, bin_edges = self.histogram

        widths = bin_edges[1:] - bin_edges[:-1]

        pdf = hist / np.sum(hist * widths)

        self.pdf = np.hstack([0.0, pdf, 0.0])
        if self.return_min:
            self.pdf[self.pdf <= 0] = np.finfo(np.float).eps
        self.bin_edges = bin_edges

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        return self.pdf[np.searchsorted(self.bin_edges, X, side="right")]


class LODA(BaseEstimator, OutlierMixin):
    """LODA: Lightweight on-line detector of anomalies"""

    def __init__(
        self,
        n_estimators: int = 1000,
        bins: Union[int, str] = "auto",
        q: float = 0.05,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        """LODA: Lightweight on-line detector of anomalies [1]

        LODA is an ensemble of histograms on random projections.
        See Pevný, T. Loda [1] for more details.

        Arguments:
            n_estimators (int, optional): number of histograms. Defaults to 1000.
            bins (Union[int, str], optional):
                * `int` - number of equal-width bins in the given range.
                * `str` - method used to calculate bin width (`numpy.histogram_bin_edges`).
                    See `numpy.histogram` bins for more details. Defaults to "auto".
            q (float, optional):
                Quantile for compution threshold from training data scores.
                This threshold is used for ``predict`` method.
                Defaults to 0.05.
            random_state (Optional[int], optional):
                Random seed used for stochastic parts. Defaults to None.
            n_jobs (Optional[int], optional): Not implemented yet. Defaults to None.
            verbose (int, optional): Verbosity of logging. Defaults to 0.

        Attributes:
            * projections_ (numpy.ndarray, shape (n_estimators, n_features)) : Random projections
            * hists_ (List[scipy.stats.rv_histogram], shape (n_estimators,)) : Histograms
            * anom_threshold_ (float) : Treshold for `predict` function.

        References:
            1. Pevný, T. Loda: Lightweight on-line detector of anomalies. Mach Learn 102, 275–304 (2016).
            <https://doi.org/10.1007/s10994-015-5521-0>
        """
        self.n_estimators = n_estimators
        self.bins = bins
        self.q = q
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs  # TODO

        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError("LODA: n_estimators must be > 0")

        if self.q < 0 or self.q > 1:
            raise ValueError("LODA: q must be in [0; 1]")

    def _init_projections(self) -> None:
        self.projections_ = np.zeros((self.n_estimators, self._shape[1]))

        non_zero_w = np.rint(self._shape[1] * (self._shape[1] ** (-1 / 2))).astype(int)

        rnd = check_random_state(self.random_state)

        indexes = rnd.rand(self.n_estimators, self._shape[1]).argpartition(
            non_zero_w, axis=1
        )[:, :non_zero_w]

        rand_values = rnd.normal(size=indexes.shape)

        for projection, chosen_d, values in zip(
            self.projections_, indexes, rand_values
        ):
            projection[chosen_d] = values

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "LODA":
        """Fit estimator

        Args:
            X (ArrayLike): shape (n_samples, n_features). Input data
            y (Optional[ArrayLike], optional): ignored.
                Present for API consistency by convention. Defaults to None.

        Returns:
            LODA: [description]
        """
        raw_data = check_array(
            X, accept_sparse=False, dtype="numeric", force_all_finite=True
        )

        self._shape = raw_data.shape

        self._init_projections()

        w_X = X @ self.projections_.T

        self.hists_ = []
        X_prob = []

        for w_x in w_X.T:
            new_hist = HistPdf(bins=self.bins, return_min=True).fit(w_x)
            self.hists_.append(new_hist)
            prob = new_hist.predict_proba(w_x)
            X_prob.append(prob)

        X_scores = np.mean(np.log(X_prob), axis=0)

        self.anom_threshold_ = np.quantile(X_scores, self.q)

        return self

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """Anomaly scores for samples

        Average of the logarithm probabilities estimated of individual projections.
        Output is proportional to the negative log-likelihood of the sample, that
        means the less likely a sample is, the higher the anomaly value it receives[1]_.
        This score is reversed for Scikit learn compatibility.

        Args:
            X (ArrayLike): shape (n_samples, n_features). Input data
        Returns:
            np.ndarray: shape (n_samples,)
            The anomaly score of the input samples. The lower, the more abnormal.
        """
        check_is_fitted(self, attributes=["projections_", "hists_"])

        w_X = X @ self.projections_.T

        X_prob = [hist.predict_proba(w_x) for hist, w_x in zip(self.hists_, w_X.T)]

        X_scores = np.mean(np.log(X_prob), axis=0)

        return X_scores

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict if samples are outliers or not

        Samples with a score lower than ``anom_threshold_`` are considered
        to be  outliers.

        Args:
            X (ArrayLike): hape (n_samples, n_features). Input data

        Returns:
            np.ndarray: shape  (n_samples,) 1 for inlineres, -1 for outliers
        """
        check_is_fitted(self, attributes=["anom_threshold_"])

        scores = self.score_samples(X)

        return np.where(scores < self.anom_threshold_, -1, 1)
