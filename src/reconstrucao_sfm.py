import cv2
import numpy as np

def estimar_poses_e_triangulacao(kp1, kp2, matches, K):
    """
    Executa Structure from Motion básico entre dois pares de imagens.

    Args:
        kp1, kp2: Keypoints das duas imagens.
        matches: Correspondências entre as imagens.
        K: Matriz intrinseca da câmera.

    Returns:
        pontos_3d: array Nx3 com coordenadas 3D.
        R, t: pose relativa da segunda imagem.
        mask_pose: máscara dos matches usados
    """

    # Extrair pontos correspondentes.....
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.queryIdx].pt for m in matches])

    # Estimando a matriz essencial
    E, mask_E = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0, prob=0.999)

    #Recuperar pose relativa
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    print(f"Total de matches: {len(matches)}")
    print(f"Válidos: {np.sum(mask_pose)}")  

    if np.sum(mask_pose) < 8:
        raise ValueError("Triangulação falhou: poucos pontos válidos após recoverPose.")
    
    mask_pose = mask_pose.ravel().astype(bool)

    pts1_valid = pts1[mask_pose]
    pts2_valid = pts2[mask_pose]

    if len(pts1_valid) == 0 or len(pts2_valid) == 0:
        raise ValueError("Nenhum ponto válido para triangulação.")


    # Matriz de projeção da câmera 1 (identidade) # Matriz de projeção da câmera 2 (R, t)
    proj1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    proj2 = K @ np.hstack((R, t))

    proj1 = proj1.astype(np.float32)
    proj2 = proj2.astype(np.float32)

    pts1_valid = pts1_valid.T.reshape(2, -1).astype(np.float32)
    pts2_valid = pts2_valid.T.reshape(2, -1).astype(np.float32)

    print(f"Shape proj1: {proj1.shape}")  # Deve ser (3,4)
    print(f"Shape proj2: {proj2.shape}")  # Deve ser (3,4)
    print(f"Shape pts1_valid: {pts1_valid.shape}")  # Deve ser (2,N)
    print(f"Shape pts2_valid: {pts2_valid.shape}")  # Deve ser (2,N)

    # Triangular pontos
    pontos_homog = cv2.triangulatePoints(proj1, proj2, pts1_valid, pts2_valid)
    pontos_3d = (pontos_homog[:3] / pontos_homog[3]).T # converter para coordenadas cartesianas

    return pontos_3d, R, t, mask_pose