from datetime import datetime
from pathlib import Path
import shutil

from networkx import maximal_independent_set
from src import deteccao_pontos, preprocessamento, reconstruir_dense
import numpy as np
import cv2
from colorama import Fore, Back, Style, init

from src.print_utils import ColorPrinter

# def executar_pipeline_basico():
#     K = np.array([
#         [2559.68, 0, 1536],
#         [0, 2559.68, 1152],
#         [0, 0, 1]
#     ])

#     dist = np.array([-0.0204997, 0, 0, 0, 0]) # apenas k1, os demais sao 0

#     img_path = "/home/kaiooliveira/Documentos/kaio/dev/Fotogrametria/fotogrametria_aplicada/data/bruto/south-building/images/P1180141.JPG" 
#     img_saida = f"/home/kaiooliveira/Documentos/kaio/dev/Fotogrametria/fotogrametria_aplicada/data/processadas/images/IMG_{datetime.now()}__processada.JPG"

#     print(f"[INFO] Processando imagem: {img_path}")
#     img_proc = preprocessamento.processar_imagem(img_path, K, dist)
#     cv2.imwrite(img_saida, img_proc)
#     print(f"[SUCCESS] Imagem salva em: {img_saida}")

def run_pipeline_fotogrametria(origem_imgs: str, pasta_destino: str, metodo="SIFT"):
    origem = Path(origem_imgs)
    destino = Path(pasta_destino) / "images"

    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True, exist_ok=True)

    K = np.array([
        [2559.68, 0, 1536],
        [0, 2559.68, 1152],
        [0, 0, 1]
    ])
    dist = np.array([-0.0204997, 0, 0, 0, 0]) # apenas k1, os demais sao 0

    ColorPrinter.info("pré-processando imagens de {origem}...")
    for path_img in sorted(origem.glob("*.JPG")):
        ColorPrinter.info(f"> {path_img.name}")
        imagem_corrigida = preprocessamento.processar_imagem(str(path_img), K, dist)

        kp, desc = deteccao_pontos.detectar_pontos(imagem_corrigida, metodo)
        
        ColorPrinter.info(f"    - {len(kp)} pontos detectados")

        # Salvar
        out_path = destino / path_img.name
        cv2.imwrite(str(out_path), imagem_corrigida)

    ColorPrinter.info("Imagens salvas para COLMAP")
    
    
    ColorPrinter.info("Iniciando pipeline do COLMAP")
    reconstruir_dense.run_colmap_pipeline(path_base=pasta_destino, max_image_size=2000, use_exhaustive_match=True)
