import cv2
import numpy as np

from src.deteccaopontos.point_detection_model import PointDetectionModel
from src.preprocessamento.models.image_model import ImageData


class DeteccaoPontos:

    def __init__(self, image: np.ndarray, metodo: str):
        self.image = image
        self.metodo = metodo

    def detectar_pontos(self):
        if self.metodo == "SIFT":
            sift = cv2.SIFT.create()
            kp, desc = sift.detectAndCompute(self.image, None)
        elif self.metodo == "ORB":
            orb = cv2.ORB.create()
            kp, desc = orb.detectAndCompute(self.image, None)
        else:
            raise ValueError("Método de detecção inválido: use 'SIFT' ou 'ORB'.")

        return kp, desc

    @staticmethod
    def corresponder_descritores(desc1, desc2, metodo="BF"):
        """Realiza correspondência entre descritores de duas imagens"""

        if metodo == 'BF':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING if desc1.dtype == np.uint8 else cv2.NORM_L2, crossCheck=True)
            matches = matcher.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)
            return matches
        else:
            raise ValueError("Método de correspondência inválido, use 'BF'")

    @staticmethod
    def desenhar_matches(img1, kp1, img2, kp2, matches, max_matches=50):
        """Desenha correspondências entre duas imagens"""
        return cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_matches], None, flags=2)