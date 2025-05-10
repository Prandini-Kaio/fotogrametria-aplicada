import cv2

def corrigir_distorcao(imagem, matriz_intrinseca, coef_distorcao):
    """
    Corrige a distorção da lente usando os parâmetros da câmera.

    Args:
        imagem(np.array): imagem original
        matriz_intriseca (np.array): matriz 3x3 com os parâmetros da câmera.
        coef_distorcao (np.array): vetor com coeficientes de distorção.
    """

    h, w = imagem.shape[:2]
    nova_matriz, _ = cv2.getOptimalNewCameraMatrix(matriz_intrinseca, coef_distorcao, (w, h), 1, (w, h))
    imagem_corrigida = cv2.undistort(imagem, matriz_intrinseca, coef_distorcao, None, nova_matriz)

    return imagem_corrigida

def ajustar_iluminacao(imagem):
    """
    Ajusta o brilho e constraste automaticamente usando equalização do histograma

    Args:
        imagem (np.array): imagem original
    
    Returns: 
        np.array:  imagem com iluminação ajustada.
    """

    imagem_yuv = cv2.cvtColor(imagem, cv2.COLOR_BGR2YUV)
    imagem_yuv[:, :, 0] = cv2.equalizeHist(imagem_yuv[:, :, 0])
    imagem_ajustada = cv2.cvtColor(imagem_yuv, cv2.COLOR_YUV2BGR)

    return imagem_ajustada

def redimensionar_imagem(imagem, largura:800):
    """
    Redimensiona a imagem proporcionalmente para acelerar o processamento

    Args:
        imagem (np.array): imagem original.
        largura (int): nova largura da imagem.
    Returns:
        np.array: imagem redimensionada
    """

    h, w = imagem.shape[:2]
    proporcao = largura / float(w)
    altura = int(h * proporcao)

    return cv2.resize(imagem, (largura, altura))

def processar_imagem(caminho_imagem, matriz_intrinseca, coef_distorcao):
    """
    Pipeline de pré-processamento: distorção + iluminação  redimensionamento.

    Args:
        caminho_imagem (str): caminho da imagem.
        matriz_intriseca (np.array): matriz da câmera.
        coef_distorcao (nj.array): coeficientes da lente.

    Returns: 
        np.array: imagem processada.
    """

    imagem = cv2.imread(caminho_imagem)
    imagem = corrigir_distorcao(imagem, matriz_intrinseca, coef_distorcao)
    imagem = ajustar_iluminacao(imagem)
    imagem = redimensionar_imagem(imagem, 800)

    return imagem