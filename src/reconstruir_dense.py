import os
import subprocess
from pathlib import Path

from src.print_utils import ColorPrinter

def run_colmap_pipeline(path_base="colmap_pipeline", max_image_size=2000, use_exhaustive_match=True):

    # Set QT environment variable para prenvinir GUI issues
    os.environ["QT_QPA_PLATAFORM"] = 'offscreen'
    os.environ["QT_DEBUG_PLUGINS"] = "0"

    path_base = Path(path_base)
    images = path_base / "images"
    sparse = path_base / "sparse"
    dense = path_base / "dense"
    database = path_base / "database.db"

    sparse.mkdir(parents=True, exist_ok=True)
    dense.mkdir(parents=True, exist_ok=True)

    def run(cmd):
        ColorPrinter(f"[COLMAP] Executando: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    
    # Extrair caracteristicas
    run([
        "colmap", "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(images),
        "--ImageReader.single_camera", "1"
    ])

    # Match entre imagens
    matcher = "exhaustive_matcher" if use_exhaustive_match else "sequential_matcher"
    run([
        "colmap", matcher,
        "--database_path", str(database)
    ])

    # Reconstrução SFM(esparsa)......
    run([
        "colmap", "mapper",
        "--database_path", str(database),
        "--image_path", str(images),
        "--output_path", str(sparse)
    ])

    # Undistort
    run([
        "colmap", "image_undistorter",
        "--image_path", str(images),
        "--input_path", str(sparse / "0"),
        "--output_path", str(dense),
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size)
    ])

    # PatchMatch stereo
    run([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(dense),
        "--worspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(dense / "fused.ply")
    ])

    ColorPrinter.success("[COLMAP] Pipeline finalizado com sucesso!")

