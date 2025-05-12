import numpy as np
import cv2
from src import deteccao_pontos

def gerar_imagem_mockada():
    """
    Cria uma imagem sintética com padrões para detecção de pontos.
    """
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 20, (255, 255, 255), -1)
    cv2.rectangle(img, (200, 200), (250, 250), (255, 255, 255), -1)
    cv2.line(img, (300, 300), (350, 350), (255, 255, 255), 5)
    return img

def test_deteccao_pontos_orb():
    img = gerar_imagem_mockada()
    kp, desc = deteccao_pontos.detectar_pontos(img, metodo="ORB")

    assert isinstance(kp, (list, tuple)), "Keypoints não é uma sequência (lista ou tupla)"
    assert desc is not None, "Descritores retornaram None."
    assert len(kp) > 0, "Nenhum ponto detectado com ORB."
    assert desc.shape[0] == len(kp), "Número de descritores e keypoints não corresponde."

def test_deteccao_pontos_sift():
    img = gerar_imagem_mockada()
    kp, desc = deteccao_pontos.detectar_pontos(img, metodo="SIFT")

    assert isinstance(kp, (list, tuple)), "Keypoints não é uma sequência (lista ou tupla)"
    assert desc is not None, "Descritores retornaram None."
    assert len(kp) > 0, "Nenhum ponto detectado com SIFT."
    assert desc.shape[0] == len(kp), "Número de descritores e keypoints não corresponde."

# Matching com descritores ORB-like
def test_correspondendia_descritores_orb():
    from src.deteccao_pontos import corresponder_descritores

    # Simula descritores de ORB (uint8)
    desc1 = np.random.randint(0, 256, (100, 32), dtype=np.uint8)
    desc2 = np.copy(desc1) # Garante o match perfeito rs

    matches = corresponder_descritores(desc1, desc2, metodo="BF")

    assert isinstance(matches, list), "Matches deve ser uma lista."
    assert all(isinstance(m, cv2.DMatch) for m in matches), "Itens não sao DMatch."
    assert len(matches) == 100, "Número incorreto de matches."
    assert matches[0].distance <= matches[-1].distance, "Matches não estão ordenados por distância."


# Matching com descritores SIFT-like
def test_correspondendia_descritores_sift():
    from src.deteccao_pontos import corresponder_descritores

    # Simula descritores de SIFT (float32)
    desc1 = np.random.rand(50, 128).astype(np.float32)
    desc2 = np.copy(desc1)

    matches = corresponder_descritores(desc1, desc2, metodo="BF")

    assert isinstance(matches, list), "Matches deve ser uma lista."
    assert all(isinstance(m, cv2.DMatch) for m in matches), "Itens não sao DMatch."
    assert len(matches) == 50, "Número incorreto de matches."
