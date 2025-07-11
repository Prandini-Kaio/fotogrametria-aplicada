from typing import Tuple

import cv2
import numpy as np

from src.preprocessamento.models.image_model import ImageData


class PreprocessamentoService:
    def __init__(self, image_data: ImageData):
        self.image_data = image_data

    def get_imagem_corrigida(self) -> ImageData:
        self.load_image()

        self.image_data.imagem_processada = self.corrigir_distorcao()
        self.image_data.imagem_processada = self.ajustar_iluminacao(self.image_data.imagem_processada)
        self.image_data.imagem_processada = self.redimensionar_imagem(self.image_data.imagem_processada, 800)

        return self.image_data


    def load_image(self) -> None:
        """Carrega a imagem e suas informações para a memôria"""

        self.image_data.imagem_original = cv2.imread(self.image_data.path)

        if(self.image_data.imagem_original is None):
            raise ValueError(f"Falha ao carregar imagem: {self.image_data.path}")

        self.image_data.dimensoes_originais = self.image_data.imagem_original.shape[:2]

    def corrigir_distorcao(self) -> np.ndarray:
        """Corrige a distorção da lente usando os parâmetros da câmera."""

        if(self.image_data.imagem_original is None):
            self.load_image()

        h, w = self.image_data.dimensoes_originais
        nova_matriz, _ = cv2.getOptimalNewCameraMatrix(
            self.image_data.matriz_intrinseca,
            self.image_data.coeficiente_distorcao,
            (w, h),
            1,
            (w, h))

        imagem_corrigida = cv2.undistort(
            self.image_data.imagem_original,
            self.image_data.matriz_intrinseca,
            self.image_data.coeficiente_distorcao,
            None,
            nova_matriz)

        return imagem_corrigida

    @staticmethod
    def ajustar_iluminacao(imagem: np.ndarray) -> np.ndarray:
        """Ajusta o brilho e constraste automaticamente usando equalização do histograma"""

        imagem_yuv = cv2.cvtColor(imagem, cv2.COLOR_BGR2YUV)
        imagem_yuv[:, :, 0] = cv2.equalizeHist(imagem_yuv[:, :, 0])
        imagem_ajustada = cv2.cvtColor(imagem_yuv, cv2.COLOR_YUV2BGR)

        return imagem_ajustada

    @staticmethod
    def redimensionar_imagem(imagem: np.ndarray, largura: int) -> np.ndarray:
        """Redimensiona a imagem proporcionalmente para acelerar o processamento"""

        h, w = imagem.shape[:2]
        proporcao = largura / float(w)
        altura = int(h * proporcao)

        return cv2.resize(imagem, (largura, altura))