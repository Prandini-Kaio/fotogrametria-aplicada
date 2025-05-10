from datetime import datetime
from src import preprocessamento
import numpy as np
import cv2

def executar_pipeline_basico():
    K = np.array([
        [2559.68, 0, 1536],
        [0, 2559.68, 1152],
        [0, 0, 1]
    ])

    dist = np.array([-0.0204997, 0, 0, 0, 0]) # apenas k1, os demais sao 0

    img_path = "/home/kaiooliveira/Documentos/kaio/dev/Fotogrametria/fotogrametria_aplicada/data/bruto/south-building/images/P1180141.JPG" 
    img_saida = f"/home/kaiooliveira/Documentos/kaio/dev/Fotogrametria/fotogrametria_aplicada/data/processadas/images/IMG_{datetime.now()}__processada.JPG"

    print(f"[INFO] Processando imagem: {img_path}")
    img_proc = preprocessamento.processar_imagem(img_path, K, dist)
    cv2.imwrite(img_saida, img_proc)
    print(f"[SUCCESS] Imagem salva em: {img_saida}")