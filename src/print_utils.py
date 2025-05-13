from colorama import Fore, Back, Style, init
from datetime import datetime
import sys

init(autoreset=True)

class ColorPrinter:
    """
    Classe para prints coloridos a nivel de log.
    Exemplos de uso:
        ColorPrinter.info("Mensagem informativa")
        ColorPrinter.warning("Aviso importante")
        ColorPrinter.success("Operação concluída")
        ColorPrinter.error("Erro crítico", exit_code=1)
    """

    @staticmethod
    def _get_timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def info(message, bold=False):
        style = Style.BRIGHT if bold else Style.NORMAL
        print(f"{Style.DIM}{ColorPrinter._get_timestamp()}{Style.RESET_ALL} {Fore.CYAN}{style}[INFO]{Style.RESET_ALL} {message}")

    @staticmethod
    def warning(message, bold=True):
        style = Style.BRIGHT if bold else Style.NORMAL
        print(f"{Style.DIM}{ColorPrinter._get_timestamp()}{Style.RESET_ALL} {Fore.YELLOW}{style}[WARNING]{Style.RESET_ALL} {message}")

    @staticmethod
    def success(message, bold=True):
        style = Style.BRIGHT if bold else Style.NORMAL
        print(f"{Style.DIM}{ColorPrinter._get_timestamp()}{Style.RESET_ALL} {Fore.GREEN}{style}[SUCCESS]{Style.RESET_ALL} {message}")

    @staticmethod
    def error(message, bold=True, exit_code=None):
        style = Style.BRIGHT if bold else Style.NORMAL
        print(f"{Style.DIM}{ColorPrinter._get_timestamp()}{Style.RESET_ALL} {Fore.RED}{style}[ERROR]{Style.RESET_ALL} {message}", file=sys.stderr)
        sys.exit(exit_code)