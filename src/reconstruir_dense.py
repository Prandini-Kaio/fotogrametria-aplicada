import os
import subprocess
from pathlib import Path

from src.print_utils import ColorPrinter

def run_colmap_pipeline(path_base="colmap_pipeline", max_image_size=2000, use_exhaustive_match=True):
    path_base = Path(path_base)
    images = path_base / "images"
    sparse = path_base / "sparse"
    dense = path_base / "dense"
    database = path_base / "database.db"

    sparse.mkdir(parents=True, exist_ok=True)
    dense.mkdir(parents=True, exist_ok=True)
    
    # Extrair caracteristicas
    extrair_caracteristicas(database, images)

    # Match entre imagens
    match_images(database, use_exhaustive_match)

    # Reconstrução SFM(esparsa)......
    reconstruir_sfm(database, images, sparse)

    # Undistort
    undistort(images, sparse, dense, max_image_size)

    # PatchMatch stereo
    patchmatch_stereo(dense)

    ColorPrinter.success("[COLMAP] Pipeline finalizado com sucesso!")

def extrair_caracteristicas(database, images):
    run([
        "colmap", "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(images),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "0"
    ])

def match_images(database, use_exhaustive_match):
    matcher = "exhaustive_matcher" if use_exhaustive_match else "sequential_matcher"
    run([
        "colmap", matcher,
        "--database_path", str(database),
        "--SiftMatching.use_gpu", "0"
    ])

def reconstruir_sfm(database, images, sparse):
    run([
        "colmap", "mapper",
        "--database_path", str(database),
        "--image_path", str(images),
        "--output_path", str(sparse)
    ])

def undistort(images, sparse, dense, max_image_size):
    run([
        "colmap", "image_undistorter",
        "--image_path", str(images),
        "--input_path", str(sparse / "0"),
        "--output_path", str(dense),
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size)
    ])

def patchmatch_stereo(dense):
    run([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--SiftExtraction.use_gpu", "0",
        "--input_type", "geometric",
        "--output_path", str(dense / "fused.ply")
    ])

def run(cmd):
    ColorPrinter.info(f"[COLMAP] Executando: {''.join(cmd)}")
    full_env = os.environ.copy()
    full_env['CUDA_PATH'] = '/usr/local/cuda'
    full_env['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + full_env.get('LD_LIBRARY_PATH', '')
    full_env["QT_QPA_PLATFORM"] = "offscreen" 
    subprocess.run(cmd, check=True, env=full_env)