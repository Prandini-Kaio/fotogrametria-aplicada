from src import Main

if __name__ == '__main__':
    Main.run_pipeline_fotogrametria("data/bruto/south-building/images", "colmap_pipeline", metodo="SIFT")
