import pytest
import numpy as np

def imagem_mock_grande():
    """
    Retorna uma imagem RGB falsa de 1600x1200 pixels, totalmente branca.
    """

    h, w = 1200, 1600
    return np.ones((h, w, 3), dtype=np.unit8) * 255
