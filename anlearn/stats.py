from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ._typing import ArrayLike


class IQR(BaseEstimator, OutlierMixin):
    """Interquartile range

    Outlier deteciton method using Tukey's fences.
    If lower quantile is 0.25 (:math:`Q_1` lower quartile) and
    upper quantile is 0.75 (:math:`Q_3` upper quartile),
    then outlier is any observation outside the range:

    .. math::
        [Q_1 - k(Q_3 - Q_1); Q_3 + k(Q_3 - Q_1)]

    John Tukey proposed :math:`k=1.5` is an outlier, and :math:`k=3` is far out.

    Parameters
    ----------
    k : float, optional
        Outlier threshold, by default 1.5
    lower_quantile : float, optional
        Lower quantile, from (0; 1), by default 0.25
    upper_quantile : float, optional
        Upper quantile, from (0; 1), by default 0.75
    ensure_2d : bool, optional
        Frobid input 1D arrays, by default True

    Attributes
    ----------
    lqv_ : float
        Lower quantile value estimated from the input data
    uqv_ : float
        Upper quantile value estimated from the input data
    iqr_ : float
        Interquartile range, :attr:`uqv_` - :attr:`lqv_`

    Example
    -------
    >>> import numpy as np
    >>> from anlearn.stats import IQR
    >>> X = np.hstack([[-7,-4], np.arange(5), [10, 15]])
    >>> iqr = IQR(ensure_2d=False)
    >>> iqr.fit(X)
    IQR(ensure_2d=False)
    >>> iqr.predict(X)
    array([-1,  1,  1,  1,  1,  1,  1,  1, -1])
    >>> iqr.score_samples(X)
    array([-1.75, -1.  , -0.  , -0.  , -0.  , -0.  , -0.  , -1.5 , -2.75])

    Raises
    ------
    ValueError
        Lower quantile must be lower than upper quantile.
    """

    def __init__(
        self,
        k: float = 1.5,
        lower_quantile: float = 0.25,
        upper_quantile: float = 0.75,
        ensure_2d: bool = True,
    ) -> None:

        self.k = k
        self.ensure_2d = ensure_2d
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

        if lower_quantile >= upper_quantile:
            raise ValueError("IQR: Lower quantile must be lower than upper quantile.")

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "IQR":
        """Fit estimator

        Parameters
        ----------
        X : ArrayLike
            Input data of shape (n_samples, 1) or (n_samples,) if `ensure_2d` is False
        y : Optional[ArrayLike], optional
            Ignored, present for API consistency by convention, by default None

        Returns
        -------
        IQR
            Fitted estimator
        """
        raw_data = check_array(
            X, force_all_finite=True, ensure_2d=self.ensure_2d
        ).flatten()

        self.lqv_, self.uqv_ = np.quantile(raw_data, (0.25, 0.75))
        self.iqr_ = self.uqv_ - self.lqv_

        return self

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """Score samples

        Score is comuputed as distance from interval :math:`[Q_{lower}; Q_{upper}]` divided
        by interquartile range. :math:`score = distance(data, (lqv, uqv)) / iqr`.
        Score is inverted for scikit-learn compatibility

        Parameters
        ----------
        X : ArrayLike
            Input data of shape (n_samples, 1) or (n_samples,) if ``ensure_2d`` is False

        Returns
        -------
        numpy.ndarray
            Shape (n_samples,). The outlier score of the input samples.
            The lower, the more abnormal.
        """
        check_is_fitted(self, attributes=["lqv_", "uqv_", "iqr_"])

        raw_data = check_array(
            X, force_all_finite=True, ensure_2d=self.ensure_2d
        ).flatten()

        scores = np.zeros(shape=raw_data.shape[0])

        l_lqv = raw_data < self.lqv_
        scores[l_lqv] = (raw_data[l_lqv] - self.lqv_) / self.iqr_

        g_uqv = raw_data > self.uqv_
        scores[g_uqv] = (raw_data[g_uqv] - self.uqv_) / self.iqr_

        return -np.abs(scores)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict if samples are outliers or not

        Samples with a score lower than ``k`` are considered to be  outliers.

        Parameters
        ----------
        X : ArrayLike
            Input data, shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray
            Shape (n_samples,) 1 for inlineres, -1 for outliers
        """
        scores = self.score_samples(X)

        return np.where(scores < -self.k, -1, 1)
