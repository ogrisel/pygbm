from pygbm.gradient_boosting import GradientBoostingMachine


# Workaround: https://github.com/numba/numba/issues/3341
import numba
numba.config.THREADING_LAYER = 'workqueue'

__version__ = '0.1.0.dev0'
__all__ = ['GradientBoostingMachine']
