"""Small extensions to anemoi-graphs/anemoi-training needed for aiLand.

Not a replacement for anything in the hand-rolled pipeline -- this is glue
code the anemoi-based pipeline (configs/anemoi/, docs/ANEMOI.md) needs that
upstream doesn't provide.
"""

import numpy as np

from anemoi.graphs.nodes.attributes.masks import BaseAnemoiDatasetVariable


class ThresholdAnemoiDatasetVariable(BaseAnemoiDatasetVariable):
    """Boolean mask of an Anemoi dataset variable exceeding a threshold.

    anemoi-graphs ships ``NonzeroAnemoiDatasetVariable`` (``!= 0``) and
    ``NonmissingAnemoiDatasetVariable`` (``not NaN``), but not a thresholded
    version. aiLand needs one: the O96 land set (11,538 points) is defined by
    ``lsm_0 > 0.5``, exactly matching ``ailand.extract``'s ``--lsm-threshold``
    default -- not "any nonzero land fraction", which would keep far more
    coastal/mixed cells than the paper's own land set.
    """

    def __init__(self, variable: str, threshold: float = 0.5) -> None:
        super().__init__(variable)
        self.threshold = threshold

    def _get_mask(self, ds: np.ndarray) -> np.ndarray:
        return ds > self.threshold
