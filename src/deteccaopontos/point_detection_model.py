from dataclasses import dataclass
from typing import Sequence

import numpy as np
from cv2 import KeyPoint


@dataclass
class PointDetectionModel:
    """Classe que representa um ponto."""
    kp: Sequence[KeyPoint]
    descriptors: np.ndarray