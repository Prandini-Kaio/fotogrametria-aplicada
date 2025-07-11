from pathlib import Path
import shutil

from src import preprocessamento
from src.dense import reconstruir_dense
from src.deteccaopontos import deteccao_pontos
import numpy as np
import cv2

from src.utils.log import Log

def run_pipeline_fotogrametria(origem_imgs: str, pasta_destino: str, metodo="SIFT"):
    origem = Path(origem_imgs)
    destino = Path(pasta_destino) / "images"

    # Cria a pasta caso nao exista
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True, exist_ok=True)

    K = np.array([
        [2559.68, 0, 1536],
        [0, 2559.68, 1152],
        [0, 0, 1]
    ])
    dist = np.array([-0.0204997, 0, 0, 0, 0]) # apenas k1, os demais sao 0

    Log.info("Iniciando pré-processamento de imagens com origem em {origem}...")
    for path_img in sorted(origem.glob("*.JPG")):
        Log.info(f"> {path_img.name}")
        imagem_corrigida = preprocessamento.processar_imagem(str(path_img), K, dist)

        kp, desc = deteccao_pontos.detectar_pontos(imagem_corrigida, metodo)
        
        Log.info(f"    - {len(kp)} pontos detectados")

        # Salvar
        out_path = destino / path_img.name
        cv2.imwrite(str(out_path), imagem_corrigida)

    Log.info("Imagens salvas para COLMAP")
    
    
    Log.info("Iniciando pipeline do COLMAP")
    reconstruir_dense.run_colmap_pipeline(path_base=pasta_destino, max_image_size=2000, use_exhaustive_match=True)
