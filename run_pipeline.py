from src import Main
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline de Fotogrametria')

    parser.add_argument(
        '--processar-imagens',
        dest='processar_imagens',
        action='store_true',
        help='Processar imagens do zero. Se não especificado, usa imagens existentes em resources/output/images'
    )

    parser.add_argument(
        '--metodo',
        type=str,
        default='SIFT',
        choices=['SIFT', 'ORB'],
        help='Método de detecção de pontos (padrão: SIFT)'
    )

    args = parser.parse_args()

    main = Main(
        "resources/input/brute-images/south-building/images",
        "resources/output",
        metodo=args.metodo
    )

    # Ensure processar_imagens is always available (defaults to False if not set)
    processar_imagens = getattr(args, 'processar_imagens', False)
    main.run_pipeline_fotogrametria(processar_imagens)
