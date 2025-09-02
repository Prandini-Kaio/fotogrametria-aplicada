from src import Main

if __name__ == '__main__':
    main = Main("resources/input/brute-images/south-building/images", "resources/output/images", metodo="SIFT")
    main.run_pipeline_fotogrametria()
