from pathlib import Path
import shutil
from typing import List

from src.dense import reconstruir_dense
from src.deteccaopontos import deteccao_pontos
import numpy as np

from src.preprocessamento.models.image_model import ImageData
from src.preprocessamento.service.preprocessamento_service import PreprocessamentoService
from src.utils.log import Log

class Main:
    def __init__(self):
        Log.info("Teste", True)

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

        Log.info(f'Iniciando pré-processamento de imagens com origem em {origem}...')
        imagens_processadas: List = []

        brute_images = sorted(origem.glob("*.JPG"))
        num_images_processed: int = 1
        for path_img in brute_images:

            Log.info(f'Processing... {(num_images_processed/len(brute_images))*100:.2f}%')
            image_data = ImageData(
                path=str(path_img),
                matriz_intrinseca=K,
                coeficiente_distorcao=dist
            )

            preprocessamento = PreprocessamentoService(image_data)
            imagens_processadas.append(preprocessamento.get_imagem_corrigida())
            num_images_processed = num_images_processed + 1

        Log.success(f'Imagens processadas: {len(imagens_processadas)}')
            # kp, desc = deteccao_pontos.detectar_pontos(imagem_corrigida, metodo)
            #
            # Log.info(f"    - {len(kp)} pontos detectados")
            #
            # # Salvar
            # out_path = destino / path_img.name
            # cv2.imwrite(str(out_path), imagem_corrigida)

        # Log.info("Imagens salvas para COLMAP")
        #
        #
        # Log.info("Iniciando pipeline do COLMAP")
        # reconstruir_dense.run_colmap_pipeline(path_base=pasta_destino, max_image_size=2000, use_exhaustive_match=True)
