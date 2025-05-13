# import numpy as np
# import cv2
# import pytest
# from src import deteccao_pontos, preprocessamento, reconstrucao_sfm

# # Objetivo:
# # - Verificar se a função retorna
# #   - Uma nuvem de pontos 3D ((N, 3))
# #   - A matriz de rotação R com forma (3, 3)
# #   - O vetor de translação t com forma (3, 1)
# # Validar:
# #   - Os pontos triangulados são finitos
# #   - Há pelo menos um ponto válido
# #   - Não há exceção se entradas forem válidas


# def gerar_par_imagem_simulada(deslocamento_px=5):
#     """
#     Cria duas imagens artificiais com formas simples e uma leve tranlação.
#     """

#     img1 = np.zeros((400, 400, 3), dtype=np.uint8)
#     cv2.circle(img1, (100, 100), 20, (255, 255, 255), -1)
#     cv2.rectangle(img1, (200, 200), (250, 250), (255, 255), -1)

#     M = np.float32([[1, 0, deslocamento_px], [0, 1, 0]])
#     img2 = cv2.warpAffine(img1, M, (400, 400))

#     return img1, img2

# def test_estimar_poses_e_triangulacao_com_mock():
#     img1, img2 = gerar_par_imagem_simulada()

#     K = np.array([
#         [800, 0, 200],
#         [0, 800, 200],
#         [0,   0,   1]
#     ], dtype=np.float32)

#     kp1, desc1 = deteccao_pontos.detectar_pontos(img1, metodo="SIFT")
#     kp2, desc2 = deteccao_pontos.detectar_pontos(img2, metodo="SIFT")

#     if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
#         pytest.skip("Descritores ou keypoints não foram gerados — imagem mockada insuficiente.")

#     matches = deteccao_pontos.corresponder_descritores(desc1, desc2, metodo="BF")

#     matches_filtrados = [
#         m for m in matches
#         if m.queryIdx < len(kp1) and m.trainIdx < len(kp2)
#     ]

#     if len(matches_filtrados) < 8:
#         pytest.skip("Poucos matches detectados — SIFT falhou em imagem mockada")
    
#     pontos_3d, R, t, _ = reconstrucao_sfm.estimar_poses_e_triangulacao(kp1, kp2, matches_filtrados, K)

#     assert pontos_3d.shape[1] == 3, "Pontos devem ter shape (N, 3)."
#     assert R.shape == (3, 3), "Matriz de rotação com shape incorreto."
#     assert t.shape == (3, 1), "Vetor de translação com shape incorreto."
#     assert np.isfinite(pontos_3d).all(), "Há valores infinitos ou NaN nos pontos 3D."
#     assert pontos_3d.shape[0] >= 5, "Poucos pontos triangulados."