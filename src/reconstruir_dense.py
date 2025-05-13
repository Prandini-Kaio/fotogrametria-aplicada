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

    def run(cmd):
        ColorPrinter.info(f"[COLMAP] Executando: {' '.join(cmd)}")
        env = os.environ.copy()
        env["QT_QPA_PLATAFORM"] = "offscreen"
        subprocess.run(cmd, check=True, env=env)
    
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
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.enable_gpu", "0",
        "--input_type", "geometric",
        "--output_path", str(dense / "fused.ply")
    ])

    ColorPrinter.success("[COLMAP] Pipeline finalizado com sucesso!")

