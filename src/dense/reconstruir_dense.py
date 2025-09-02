import os
import subprocess
from pathlib import Path
from shlex import shlex

from src.utils.log import Log


class DenseService:

    def __init__(self, base_path: str, max_image_size: int, use_exhaustive_match: bool):
        self.base_path = Path(base_path)
        self.max_image_size = max_image_size
        self.use_exhaustive_match = use_exhaustive_match

    def gerar_caminhos(self):
        self.images = self.base_path / "imagens"
        self.sparse = self.base_path / "sparse"
        self.dense = self.base_path / "dense"
        self.database = self.base_path / "database"

        self.sparse.mkdir(parents=True, exist_ok=True)
        self.dense.mkdir(parents=True, exist_ok=True)



    def iniciar_processamento(self):

        self.gerar_caminhos()

        self.extrair_caracteristicas()

        self.match_images()

        # Reconstrução SFM(esparsa)......
        self.reconstruir_sfm()

        # Undistort
        self.undistort()

        # PatchMatch stereo
        self.patchmatch_stereo()

        Log.success("[COLMAP] Pipeline finalizado com sucesso!")

    def extrair_caracteristicas(self):
        self.run([
            "colmap", "feature_extractor",
            "--database_path", str(self.database),
            "--image_path", str(self.images),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "0"
        ])

    def match_images(self):
        matcher = "exhaustive_matcher" if self.use_exhaustive_match else "sequential_matcher"
        self.run([
            "colmap", matcher,
            "--database_path", str(self.database),
            "--SiftMatching.use_gpu", "0"
        ])

    def reconstruir_sfm(self):
        self.run([
            "colmap", "mapper",
            "--database_path", str(self.database),
            "--image_path", str(self.images),
            "--output_path", str(self.sparse)
        ])

    def undistort(self):
        self.run([
            "colmap", "image_undistorter",
            "--image_path", str(self.images),
            "--input_path", str(self.sparse / "0"),
            "--output_path", str(self.dense),
            "--output_type", "COLMAP",
            "--max_image_size", str(self.max_image_size)
        ])

    def patchmatch_stereo(self):
        self.run([
            "colmap", "patch_match_stereo",
            "--workspace_path", str(self.dense),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true",
            "--SiftExtraction.use_gpu", "0",
            "--input_type", "geometric",
            "--output_path", str(self.dense) /  "fused.ply"
        ])

    @staticmethod
    def run(cmd):
        Log.info(f'[COLMAP] Executando: {"".join(cmd)}')
        full_env = os.environ.copy()
        full_env['CUDA_PATH'] = '/usr/local/cuda'
        full_env['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + full_env.get('LD_LIBRARY_PATH', '')
        full_env["QT_QPA_PLATFORM"] = "offscreen"
        subprocess.run(cmd, check=True, env=full_env)