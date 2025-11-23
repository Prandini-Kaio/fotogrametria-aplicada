import shutil
import uuid
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from src.dense import reconstruir_dense
from src.dense.reconstruir_dense import DenseService
from src.deteccaopontos import deteccao_pontos
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
        output_path = self.output_path / "images"
        if output_path.exists():
            Log.warning("Removendo imagens criadas anteriormente.")
            shutil.rmtree(self.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # TODO -> ALTERAR MATRIZ FIXA DA CAMERA E DISTORCAO
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
            imagem_processada = preprocessamento.get_imagem_corrigida()

            imagens_processadas.append(imagem_processada)
            Main._salvar_imagem(imagem_processada, output_path)

            _num_images_processed = _num_images_processed + 1

        Log.success(f'Imagens processadas: {len(imagens_processadas)}')
        return imagens_processadas

    def ler_imagens_processadas(self) -> List[ImageData]:
        """
        Lê imagens já processadas da pasta output e cria objetos ImageData
        """
        output_path = self.output_path / "images"

        Log.info(f'Lendo imagens processadas de {output_path}...')

        if not output_path.exists():
            Log.error(f'Diretório {output_path} não existe!')
            return []

        # TODO -> ALTERAR MATRIZ FIXA DA CAMERA
        matriz_k = np.array([
            [2559.68, 0, 1536],
            [0, 2559.68, 1152],
            [0, 0, 1]
        ])
        distorcao = np.array([-0.0204997, 0, 0, 0, 0])  # apenas k1, os demais sao 0

        imagens_processadas: List[ImageData] = []
        processed_images = sorted(output_path.glob("*.png"))

        if not processed_images:
            Log.warning(f'Nenhuma imagem PNG encontrada em {output_path}')
            return []

        _num_images_loaded: int = 1
        for path_img in processed_images:
            Log.info(f'Loading... {(_num_images_loaded / len(processed_images)) * 100:.2f}%')

            # Carregar a imagem
            imagem = np.array(Image.open(path_img))

            # Criar objeto ImageData
            image_data = ImageData(
                path=str(path_img),
                matriz_intrinseca=matriz_k,
                coeficiente_distorcao=distorcao,
                imagem_processada=imagem,
                dimensoes_processadas=np.array(imagem.shape[:2])
            )

            imagens_processadas.append(image_data)
            _num_images_loaded += 1

        Log.success(f'Imagens carregadas: {len(imagens_processadas)}')
        return imagens_processadas

    @staticmethod
    def detectar_pontos(imagens_processadas: List[ImageData], metodo: str, output_path: Path) -> List[PointDetectionModel]:
        Log.info(f'Iniciando deteccao de pontos para {len(imagens_processadas)} imagens...')

        points: List[PointDetectionModel] = []
        for data in imagens_processadas:

            deteccao_pontos = DeteccaoPontos(data.imagem_processada, metodo)
            keypoints, descritor = deteccao_pontos.detectar_pontos()

            points.append(PointDetectionModel(keypoints, descritor))
            Log.info(f"    - {len(keypoints)} pontos detectados")

        Log.success(f'Imagens processadas: {len(points)}')
        return points

    @staticmethod
    def _salvar_imagem(imagem: ImageData, output_path):
        image_array = Image.fromarray(imagem.imagem_processada)

        nome_arquivo = f"{output_path}/{uuid.uuid4()}.png"

        image_array.save(nome_arquivo, "PNG")
        Log.info(f'Imagem salva em {nome_arquivo}.')

    @staticmethod
    def gerar_dense(path):
        dense: DenseService = DenseService(base_path=path, max_image_size=2000, use_exhaustive_match=True)
        dense.iniciar_processamento()

    def run_pipeline_fotogrametria(self, processar_imagens: bool):
        if processar_imagens:
            imagens_processadas = self.preprocessar_imagens()
        else:
            self._verificar_diretorio_de_entrada()
            imagens_processadas = self.ler_imagens_processadas()

        self.detectar_pontos(imagens_processadas, self.metodo, self.output_path)
        self.gerar_dense(self.output_path)

        Log.success('Processamento concluído!')

    def _verificar_diretorio_de_entrada(self):
        image_output = self.output_path / "images"
        if not image_output or not list(image_output.glob("*.png")):
            Log.error(f'O diretorio {image_output} não existe ou está vazio')
            raise Exception('Diretorio de saida vazio')

