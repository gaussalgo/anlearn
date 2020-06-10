from pkg_resources import DistributionNotFound, get_distribution

from .loda import LODA
from .stats import IQR

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass


__all__ = ["LODA", "IQR"]
