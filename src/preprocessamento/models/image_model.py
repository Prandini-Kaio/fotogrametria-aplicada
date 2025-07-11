from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ImageData:
    """Represent the data of imagfe and your metadatas"""
    path: str
    matriz_intrinseca: np.ndarray
    coeficiente_distorcao: np.ndarray
    imagem_original: Optional[np.ndarray] = None
    imagem_processada: Optional[np.ndarray] = None

    dimensoes_originais: Optional[np.ndarray] = None
    dimensoes_processadas: Optional[np.ndarray] = None

    @property
    def nome_arquivo(self) -> str:
        return self.path.split('/')[-1]
