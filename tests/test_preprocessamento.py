import numpy as np
import cv2
import os
from src import preprocessamento

def test_processar_imagem_valida():
    path = "/home/kaiooliveira/Documentos/kaio/dev/Fotogrametria/fotogrametria_aplicada/data/bruto/south-building/images/P1180141.JPG" 
    assert os.path.exists(path), f"Imagem de teste não encontrada {path}"

    # Paramêtros da câmera
    K = np.array([
        [2559.68, 0, 1536],
        [0, 2559.68, 1152],
        [0, 0, 1]
    ], dtype=np.float32)

    dist = np.array([-0.0204997, 0, 0, 0, 0])


    img = preprocessamento.processar_imagem(path, K, dist)

    assert img is not None, "Imagem retornada é None"
    assert isinstance(img, np.ndarray), "imagem não é um array NumPy"
    assert img.ndim == 3, "Imagem não tem 3 dimensões (altura, largura, canais)"
    assert img.shape[2] == 3, "Imagem não tem 3 canais (RGB)"

def test_redimensionar_imagem_mockada(imagem_mock_grande):
    from src.preprocessamento import redimensionar_imagem

    largura_esperada = 800
    nova_img = redimensionar_imagem(imagem_mock_grande, largura=largura_esperada)

    assert nova_img.shape[1] == largura_esperada, f"Largura incorreta: {nova_img.shape[1]}"
    assert nova_img.shape[0] == 600, f"Altura proporcional incorreta: {nova_img.shape[0]}"
    assert nova_img.shape[2] == 3, "Imagem deve ter 3 canais (RGB)"