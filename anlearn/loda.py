from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from ._typing import ArrayLike


# There is no DensityEstimationMixin in scikit-learn
class Histogram(BaseEstimator):
    """Histogram model

    Histogram model based on :obj:`scipy.stats.rv_histogram`.

    Parameters
    ----------
    bins : Union[int, str], optional
        * :obj:`int` - number of equal-width bins in the given range.
        * :obj:`str` - method used to calculate bin width (:obj:`numpy.histogram_bin_edges`).

        See :obj:`numpy.histogram_bin_edges` bins for more details, by default "auto"
    return_min : bool, optional
        Return minimal float value instead of 0, by default True

    Attributes
    ----------
    hist : numpy.ndarray
        Value of histogram
    bin_edges : numpy.ndarray
        Edges of histogram
    pdf : numpy.ndarray
        Probability density function
    """

    def __init__(self, bins: Union[int, str] = "auto", return_min: bool = True) -> None:
        self.bins = bins
        self.return_min = return_min

    def fit(self, X: np.ndarray) -> "Histogram":
        """Fit estimator

        Parameters
        ----------
        X : numpy.ndarray
            Input data, shape (n_samples,)

        Returns
        -------
        Histogram
            Fitted estimator
        """

        self.hist, self.bin_edges = np.histogram(X, bins=self.bins)

        widths = self.bin_edges[1:] - self.bin_edges[:-1]

        pdf = self.hist / np.sum(self.hist * widths)

        self.pdf = np.hstack([0.0, pdf, 0.0])
        if self.return_min:
            self.pdf[self.pdf <= 0] = np.finfo(np.float_).eps

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability

        Predict probability of input data X.

        Parameters
        ----------
        X : numpy.ndarray
            Input data, shape (n_samples,)

        Returns
        -------
        numpy.ndarray
            Probability estimated from histogram, shape (n_samples,)
        """

        return self.pdf[np.searchsorted(self.bin_edges, X, side="right")]


class LODA(BaseEstimator, OutlierMixin):
    """LODA: Lightweight on-line detector of anomalies [1]_

    LODA is an ensemble of histograms on random projections.
    See Pevný, T. Loda [1]_ for more details.

    Parameters
    ----------
    n_estimators : int, optional
        number of histograms, by default 1000
    bins : Union[int, str], optional
        * :obj:`int` - number of equal-width bins in the given range.
        * :obj:`str` - method used to calculate bin width (:obj:`numpy.histogram_bin_edges`).

        See :obj:`numpy.histogram_bin_edges` bins for more details, by default "auto"
    q : float, optional
        Quantile for compution threshold from training data scores.
        This threshold is used for `predict` method, by default 0.05
    random_state : Optional[int], optional
        Random seed used for stochastic parts., by default None
    n_jobs : Optional[int], optional
        Not implemented yet, by default None
    verbose : int, optional
        Verbosity of logging, by default 0

    Attributes
    ----------
    projections_ : numpy.ndarray
        Random projections, shape (n_estimators, n_features)
    hists_ : List[Histogram]
        Histograms on random projections, shape (n_estimators,)
    anomaly_threshold_ : float
        Treshold for :meth:`predict` function

    Examples
    --------
    >>> import numpy as np
    >>> from anlearn.loda import LODA
    >>> X = np.array([[0, 0], [0.1, -0.2], [0.3, 0.2], [0.2, 0.2], [-5, -5], [0.6, 0.7]])
    >>> loda = LODA(n_estimators=10, bins=10, random_state=42)
    >>> loda.fit(X)
    LODA(bins=10, n_estimators=10, random_state=42)
    >>> loda.predict(X)
    array([ 1,  1,  1,  1, -1,  1])

    References
    ----------
    .. [1] Pevný, T. Loda: Lightweight on-line detector of anomalies. Mach Learn 102, 275–304 (2016).
           <https://doi.org/10.1007/s10994-015-5521-0>
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        bins: Union[int, str] = "auto",
        q: float = 0.05,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ) -> None:

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

        non_zero_w = np.rint(np.sqrt(self._shape[1])).astype(int)

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

        Parameters
        ----------
        X : ArrayLike
            Input data, shape (n_samples, n_features)
        y : Optional[ArrayLike], optional
            Present for API consistency by convention, by default None

        Returns
        -------
        LODA
            Fitted estimator
        """
        raw_data = self.__check_array(X)

        self._shape = raw_data.shape

        self._init_projections()

        w_X = raw_data @ self.projections_.T

        self.hists_ = []
        X_prob = []

        for w_x in w_X.T:
            new_hist = Histogram(bins=self.bins, return_min=True).fit(w_x)
            self.hists_.append(new_hist)
            prob = new_hist.predict_proba(w_x)
            X_prob.append(prob)

        X_scores = np.mean(np.log(np.array(X_prob)), axis=0)

        self.anomaly_threshold_ = np.quantile(X_scores, self.q)

        return self

    def __check_array(self, X: ArrayLike) -> np.ndarray:
        return check_array(
            X, accept_sparse=True, dtype="numeric", force_all_finite=True
        )

    def __log_prob(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, attributes=["projections_", "hists_"])

        raw_data = self.__check_array(X)

        w_X = raw_data @ self.projections_.T

        X_prob = np.array(
            [hist.predict_proba(w_x) for hist, w_x in zip(self.hists_, w_X.T)]
        )

        return np.log(X_prob)

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """Anomaly scores for samples

        Average of the logarithm probabilities estimated of individual projections.
        Output is proportional to the negative log-likelihood of the sample, that
        means the less likely a sample is, the higher the anomaly value it receives [1]_.
        This score is reversed for scikit-learn compatibility.

        Parameters
        ----------
        X : ArrayLike
            Input data, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray
            The anomaly score of the input samples. The lower, the more abnormal.
            Shape (n_samples,)
        """
        X_log_prob = self.__log_prob(X)

        X_scores = np.mean(X_log_prob, axis=0)

        return X_scores

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict if samples are outliers or not

        Samples with a score lower than :attr:`anomaly_threshold_` are considered
        to be  outliers.

        Parameters
        ----------
        X : ArrayLike
            Input data, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray
            1 for inlineres, -1 for outliers, shape (n_samples,)
        """
        check_is_fitted(self, attributes=["anomaly_threshold_"])

        scores = self.score_samples(X)

        return np.where(scores < self.anomaly_threshold_, -1, 1)

    def score_features(self, X: ArrayLike) -> np.ndarray:
        r"""Feature importance

        Feature importance is computed as a one-tailed two-sample t-test between
        :math:`-log(\hat{p}_i)` from histograms on projections with and without a
        specific feature. The higher the value is, the more important feature is.

        See full description in **3.3  Explaining the cause of an anomaly** [1]_ for
        more details.

        Parameters
        ----------
        X : ArrayLike
            input data, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray
            Feature importance in anomaly detection.


        Notes
        -----

        .. math::

            t_j = \frac{\mu_j - \bar{\mu}_j}{
                \sqrt{\frac{s^2_j}{|I_j|} + \frac{\bar{s}^2_j}{|\bar{I_j}|}}}


        """
        X_neg_log_prob = -self.__log_prob(X)

        zero_projections = self.projections_ == 0

        results = []
        # t-test for every feature
        for j_feature in range(self._shape[1]):
            i_with_feature = X_neg_log_prob[~zero_projections[:, j_feature]]
            i_wo_feature = X_neg_log_prob[zero_projections[:, j_feature]]

            t_j = (
                np.mean(i_with_feature, axis=0) - np.mean(i_wo_feature, axis=0)
            ) / np.sqrt(
                np.var(i_with_feature, axis=0) / i_with_feature.shape[0]
                + np.var(i_wo_feature, axis=0) / i_wo_feature.shape[0]
            )

            results.append(t_j)

        return np.vstack(results).T
