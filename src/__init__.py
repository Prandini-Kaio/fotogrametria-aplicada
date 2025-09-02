from pathlib import Path
import shutil
from typing import List

import cv2

from src.dense import reconstruir_dense
from src.dense.reconstruir_dense import DenseService
from src.deteccaopontos import deteccao_pontos
import numpy as np

from src.deteccaopontos.deteccao_pontos import DeteccaoPontos
from src.deteccaopontos.point_detection_model import PointDetectionModel
from src.preprocessamento.models.image_model import ImageData
from src.preprocessamento.service.preprocessamento_service import PreprocessamentoService
from src.utils.log import Log

class Main:
    def __init__(self, input_path: str, output_path: str, metodo: str) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.metodo = metodo

    def preprocessar_imagens(self) -> List[ImageData]:
        Log.info(f'Iniciando pré-processamento de imagens com origem em {self.input_path}...')
        if self.output_path.exists():
            Log.warning("Removendo imagens criadas anteriormente.")
            shutil.rmtree(self.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        matriz_k = np.array([
            [2559.68, 0, 1536],
            [0, 2559.68, 1152],
            [0, 0, 1]
        ])
        distorcao = np.array([-0.0204997, 0, 0, 0, 0])  # apenas k1, os demais sao 0

        imagens_processadas: List = []
        brute_images = sorted(self.input_path.glob("*.JPG"))

        _num_images_processed: int = 1
        for path_img in brute_images:

            Log.info(f'Processing... {(_num_images_processed/len(brute_images))*100:.2f}%')
            image_data = ImageData(
                path=str(path_img),
                matriz_intrinseca=matriz_k,
                coeficiente_distorcao=distorcao
            )

            preprocessamento = PreprocessamentoService(image_data)
            imagens_processadas.append(preprocessamento.get_imagem_corrigida())

            _num_images_processed = _num_images_processed + 1

        Log.success(f'Imagens processadas: {len(imagens_processadas)}')
        return imagens_processadas


    @staticmethod
    def detectar_pontos(imagens_processadas: List[ImageData], metodo: str, output_path: Path) -> List[PointDetectionModel]:
        Log.info(f'Iniciando deteccao de pontos para {len(imagens_processadas)} imagens...')

        points: List[PointDetectionModel] = []
        for data in imagens_processadas:

            deteccao_pontos = DeteccaoPontos(data.imagem_processada, metodo)
            point_data: PointDetectionModel = deteccao_pontos.detectar_pontos()

            points.append(point_data)
            Log.info(f"    - {len(point_data.kp)} pontos detectados")
            cv2.imwrite(str(output_path/data.nome_arquivo), data.imagem_processada)

        Log.success(f'Imagens processadas: {len(points)}')
        return points

    @staticmethod
    def gerar_dense():
        dense: DenseService = DenseService(base_path="resources/output/colmap_output", max_image_size=2000, use_exhaustive_match=True)
        dense.iniciar_processamento()

    def run_pipeline_fotogrametria(self):
        imagens_processadas = self.preprocessar_imagens()
        self.detectar_pontos(imagens_processadas, self.metodo, self.output_path)
        self.gerar_dense()

        Log.success('Processamento concluído!')
