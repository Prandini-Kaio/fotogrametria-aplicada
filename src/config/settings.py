from pathlib import Path
import os
import re

def load_env_file(env_path=".env"):
    """Carrega variáveis de um arquivo .env"""
    env_vars = {}
    env_file = Path(env_path)
    
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Ignorar comentários e linhas vazias
                if not line or line.startswith("#"):
                    continue
                # Processar linha no formato KEY=VALUE ou KEY="VALUE"
                match = re.match(r'^([^=]+)=(.*)$', line)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    # Remover aspas se houver
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    env_vars[key] = value
    
    return env_vars

# Carregar .env manualmente
_env_vars = load_env_file()

# Classe simples que lê do .env e variáveis de ambiente
class Settings:
    def __init__(self):
        self.APP_NAME = _env_vars.get("APP_NAME") or os.getenv("APP_NAME", "Fotogrametria aplicada")
        self.DEBUG = (_env_vars.get("DEBUG") or os.getenv("DEBUG", "False")).lower() == "true"
        self.IMAGE_STORAGE_PATH = _env_vars.get("IMAGE_STORAGE_PATH") or os.getenv("IMAGE_STORAGE_PATH", "./resources/output/images")
        # COLMAP_PATH: prioridade para .env, depois variável de ambiente
        self.COLMAP_PATH = _env_vars.get("COLMAP_PATH") or os.getenv("COLMAP_PATH", "")
        # Remover aspas se ainda houver
        if self.COLMAP_PATH:
            self.COLMAP_PATH = self.COLMAP_PATH.strip().strip('"').strip("'")