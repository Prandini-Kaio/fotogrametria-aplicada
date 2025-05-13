from src import run_pipeline_fotogrametria

if __name__ == '__main__':
    run_pipeline_fotogrametria("data/bruto/south-building/images", "colmap_pipeline", metodo="SIFT")