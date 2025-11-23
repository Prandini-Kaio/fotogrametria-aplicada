#!/usr/bin/env python3
"""
Script para iniciar o servidor da API FastAPI
"""
import uvicorn
from pathlib import Path
import sys

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    # Iniciar servidor FastAPI
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Recarregar automaticamente em desenvolvimento
        log_level="info"
    )




