import cv2
import numpy as np

def detectar_pontos(imagem, metodo="ORB"):
    """
    Detecta pontos-chave e descritores em uma imagem.

    Args:
        imagem (np.array): imagem de entrada.
        metodo (str): 'ORB' ou 'SIFT'. Defaults to "ORB".
    
        Returns:
            keypoints (list), descritores (np.array)
    """

    if metodo == "SIFT":
        sift = cv2.SIFT.create()
        kp, desc = sift.detectAndCompute(imagem, None)
    elif metodo == "ORB":
        orb = cv2.ORB.create()
        kp, desc = orb.detectAndCompute(imagem, None)
    else:
        raise ValueError("Método de detecção inválido: use 'SIFT' ou 'ORB'.")
    
    return kp, desc

def corresponder_descritores(desc1, desc2, metodo="BF"):
    """
    Realiza correspondência entre descritores de duas imagens

    Args:
        desc1, desc2 (np.array): descritores das imagens.
        metodo (str): 'BF' (Brute Force) ou 'FLANN'. Defaults to "BF".

    Returns:
        lista de correspondências.
    """

    if metodo == 'BF':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING if desc1.dtype == np.uint8 else cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    else:
        raise ValueError("Método de correspondência inválido, use 'BF'")
    
def desenhar_matches(img1, kp1, img2, kp2, matches, max_matches=50):
    """
    Desenha correspondências entre duas imagens
    """
    return cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_matches], None, flags=2)