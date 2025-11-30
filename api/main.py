from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import uuid
import json
import threading
import time
import io
import sys
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from contextlib import redirect_stdout, redirect_stderr

from src import Main
from src.dense.reconstruir_dense import DenseService

app = FastAPI(title="Fotogrametria API", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diretório de status de processamento
STATUS_DIR = Path("resources/status")
STATUS_DIR.mkdir(parents=True, exist_ok=True)

# Diretório de logs
LOGS_DIR = Path("resources/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Armazenamento de tarefas em execução (em produção, usar Redis ou banco de dados)
processing_tasks: Dict[str, Dict] = {}

# Classe para capturar logs
class LogCapture:
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.logs: List[Dict] = []
        self.log_file = LOGS_DIR / f"{task_id}.log"
        
    def write(self, message):
        if message.strip():
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "message": message.strip()
            }
            self.logs.append(log_entry)
            
            # Salvar em arquivo também
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}")
    
    def flush(self):
        pass
    
    def get_recent_logs(self, limit: int = 50) -> List[Dict]:
        return self.logs[-limit:] if len(self.logs) > limit else self.logs

class ProcessingRequest(BaseModel):
    input_path: Optional[str] = "resources/input/brute-images/south-building/images"
    output_path: Optional[str] = "resources/output"
    metodo: Optional[str] = "SIFT"
    processar_imagens: Optional[bool] = False
    gerar_malha: Optional[bool] = True  # Se deve gerar arquivos PLY

class ProcessingStatus(BaseModel):
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float
    message: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error: Optional[str] = None
    estimated_time_remaining: Optional[str] = None
    current_step: Optional[str] = None
    logs: Optional[list] = None

def calculate_estimated_time(progress: float, elapsed_time: float) -> Optional[str]:
    """Calcula tempo estimado restante baseado no progresso"""
    if progress <= 0 or elapsed_time <= 0:
        return None
    
    if progress >= 100:
        return "Concluído"
    
    total_estimated = elapsed_time / (progress / 100)
    remaining = total_estimated - elapsed_time
    
    if remaining < 60:
        return f"{int(remaining)}s"
    elif remaining < 3600:
        minutes = int(remaining / 60)
        seconds = int(remaining % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(remaining / 3600)
        minutes = int((remaining % 3600) / 60)
        return f"{hours}h {minutes}m"

def run_pipeline_async(task_id: str, request: ProcessingRequest):
    """Executa o pipeline de forma assíncrona"""
    # Importar sys no início da função para garantir que está disponível
    import sys
    
    status_file = STATUS_DIR / f"{task_id}.json"
    log_capture = LogCapture(task_id)
    start_time = time.time()
    
    # Salvar referências antes do try para garantir que estão disponíveis no finally
    old_stdout = None
    old_stderr = None
    
    try:
        processing_tasks[task_id]["status"] = "running"
        processing_tasks[task_id]["message"] = "Iniciando processamento..."
        processing_tasks[task_id]["progress"] = 0.0
        processing_tasks[task_id]["current_step"] = "Inicialização"
        processing_tasks[task_id]["logs"] = []
        
        # Redirecionar stdout e stderr para capturar logs
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = log_capture
        sys.stderr = log_capture
        
        # Atualizar arquivo de status
        update_status_file(task_id, processing_tasks[task_id])
        
        add_log(task_id, "🚀 Iniciando pipeline de fotogrametria...")
        add_log(task_id, f"📁 Caminho de entrada: {request.input_path}")
        add_log(task_id, f"📁 Caminho de saída: {request.output_path}")
        add_log(task_id, f"🔧 Método: {request.metodo}")
        
        # Inicializar pipeline
        main = Main(
            input_path=request.input_path,
            output_path=request.output_path,
            metodo=request.metodo
        )
        
        # Processar imagens se solicitado
        if request.processar_imagens:
            update_task_status(task_id, "Pré-processando imagens...", 5.0, "Pré-processamento", start_time)
            add_log(task_id, "🖼️ Iniciando pré-processamento de imagens...")
            imagens_processadas = main.preprocessar_imagens()
            add_log(task_id, f"✅ {len(imagens_processadas)} imagens pré-processadas com sucesso")
        else:
            add_log(task_id, "📂 Verificando diretório de entrada...")
            main._verificar_diretorio_de_entrada()
            add_log(task_id, "✅ Diretório de entrada verificado")
            
        # Detectar pontos
        update_task_status(task_id, "Detectando pontos de interesse...", 15.0, "Detecção de Pontos", start_time)
        add_log(task_id, f"🔍 Iniciando detecção de pontos usando {request.metodo}...")
        
        # Se não processou imagens, precisa ler as já processadas
        if not request.processar_imagens:
            add_log(task_id, "📖 Carregando imagens processadas...")
            imagens_processadas = main.ler_imagens_processadas()
            add_log(task_id, f"✅ {len(imagens_processadas)} imagens carregadas")
        
        # Detectar pontos (apenas para análise, não é obrigatório para gerar_dense)
        main.detectar_pontos(imagens_processadas, request.metodo, Path(request.output_path))
        add_log(task_id, "✅ Detecção de pontos concluída")
        
        # Gerar reconstrução densa (esta é a parte mais demorada)
        update_task_status(task_id, "Iniciando reconstrução densa (COLMAP)...", 25.0, "Reconstrução Densa", start_time)
        add_log(task_id, "🏗️ Iniciando reconstrução densa com COLMAP...")
        add_log(task_id, "⏳ Esta etapa pode levar vários minutos...")
        
        # Criar wrapper para DenseService com callbacks
        # Importar aqui para evitar importação circular
        from pathlib import Path as PathLib
        sys.path.insert(0, str(PathLib(__file__).parent.parent))
        from api.dense_wrapper import DenseServiceWrapper
        dense_wrapper = DenseServiceWrapper(main, request.output_path, task_id, start_time)
        dense_wrapper.iniciar_processamento()
        
        # Gerar malha 3D (PLY) se solicitado
        if request.gerar_malha:
            update_task_status(task_id, "Gerando malha 3D (PLY)...", 90.0, "Geração de Malha", start_time)
            add_log(task_id, "📦 Gerando arquivos PLY...")
            try:
                generate_ply_files(Path(request.output_path), task_id, start_time)
                add_log(task_id, "✅ Arquivos PLY gerados com sucesso")
            except Exception as e:
                add_log(task_id, f"⚠️ Aviso: Geração de PLY falhou: {str(e)}")
                processing_tasks[task_id]["message"] = f"Processamento concluído, mas geração de PLY falhou: {str(e)}"
                update_status_file(task_id, processing_tasks[task_id])
        
        # Finalizar
        elapsed = time.time() - start_time
        elapsed_str = format_time(elapsed)
        add_log(task_id, f"🎉 Processamento concluído com sucesso em {elapsed_str}!")
        
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["message"] = f"Processamento concluído com sucesso! Tempo total: {elapsed_str}"
        processing_tasks[task_id]["progress"] = 100.0
        processing_tasks[task_id]["current_step"] = "Concluído"
        processing_tasks[task_id]["estimated_time_remaining"] = "0s"
        processing_tasks[task_id]["end_time"] = datetime.now().isoformat()
        processing_tasks[task_id]["logs"] = log_capture.get_recent_logs(100)
        update_status_file(task_id, processing_tasks[task_id])
        
    except Exception as e:
        elapsed = time.time() - start_time
        add_log(task_id, f"❌ Erro durante processamento: {str(e)}")
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["message"] = f"Erro: {str(e)}"
        processing_tasks[task_id]["error"] = str(e)
        processing_tasks[task_id]["end_time"] = datetime.now().isoformat()
        processing_tasks[task_id]["logs"] = log_capture.get_recent_logs(100)
        update_status_file(task_id, processing_tasks[task_id])
    finally:
        # Restaurar stdout e stderr apenas se foram redirecionados
        if old_stdout is not None:
            sys.stdout = old_stdout
        if old_stderr is not None:
            sys.stderr = old_stderr

def generate_ply_files(output_path: Path, task_id: str, start_time: float):
    """Gera arquivos PLY a partir da reconstrução densa"""
    dense_path = output_path / "dense"
    
    if not dense_path.exists():
        raise FileNotFoundError(f"Diretório denso não encontrado: {dense_path}")
    
    # Procurar pelo diretório sparse dentro de dense
    sparse_dirs = list(dense_path.glob("**/sparse"))
    if not sparse_dirs:
        # Tentar usar o diretório dense diretamente
        workspace_path = dense_path
    else:
        workspace_path = sparse_dirs[0].parent
    
    # Verificar se patch_match_stereo foi executado
    stereo_dir = workspace_path / "stereo"
    if not stereo_dir.exists() or not list(stereo_dir.glob("depth_maps/*.bin")):
        raise FileNotFoundError("Reconstrução densa não concluída. Execute patch_match_stereo primeiro.")
    
    from src.dense.reconstruir_dense import DenseService
    
    # Gerar arquivo fused.ply
    update_task_status(task_id, "Gerando nuvem de pontos (fused.ply)...", 90.0, "PLY - Fusão", start_time)
    add_log(task_id, "🔗 Fundindo mapas de profundidade...")
    dense_service = DenseService(str(output_path), 2000, True)
    
    # Stereo fusion
    DenseService.run([
        "colmap", "stereo_fusion",
        "--workspace_path", str(workspace_path),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(workspace_path / "fused.ply")
    ])
    add_log(task_id, "✅ Nuvem de pontos gerada")
    
    # Gerar malha Poisson
    if (workspace_path / "fused.ply").exists():
        update_task_status(task_id, "Gerando malha Poisson...", 95.0, "PLY - Malha", start_time)
        add_log(task_id, "🔷 Gerando malha usando algoritmo Poisson...")
        DenseService.run([
            "colmap", "poisson_mesher",
            "--input_path", str(workspace_path / "fused.ply"),
            "--output_path", str(workspace_path / "meshed-poisson.ply")
        ])
        add_log(task_id, "✅ Malha Poisson gerada")
    
    update_task_message(task_id, "Arquivos PLY gerados com sucesso!")

def format_time(seconds: float) -> str:
    """Formata tempo em segundos para string legível"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"

def add_log(task_id: str, message: str):
    """Adiciona log à tarefa"""
    if task_id in processing_tasks:
        if "logs" not in processing_tasks[task_id]:
            processing_tasks[task_id]["logs"] = []
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message
        }
        processing_tasks[task_id]["logs"].append(log_entry)
        
        # Manter apenas os últimos 200 logs
        if len(processing_tasks[task_id]["logs"]) > 200:
            processing_tasks[task_id]["logs"] = processing_tasks[task_id]["logs"][-200:]
        
        update_status_file(task_id, processing_tasks[task_id])

def update_task_status(task_id: str, message: str, progress: float, step: str, start_time: float):
    """Atualiza status da tarefa com estimativa de tempo"""
    if task_id in processing_tasks:
        elapsed = time.time() - start_time
        estimated = calculate_estimated_time(progress, elapsed)
        
        processing_tasks[task_id]["message"] = message
        processing_tasks[task_id]["progress"] = progress
        processing_tasks[task_id]["current_step"] = step
        processing_tasks[task_id]["estimated_time_remaining"] = estimated
        
        update_status_file(task_id, processing_tasks[task_id])

def update_task_message(task_id: str, message: str):
    """Atualiza mensagem da tarefa"""
    if task_id in processing_tasks:
        processing_tasks[task_id]["message"] = message
        update_status_file(task_id, processing_tasks[task_id])

def update_status_file(task_id: str, status_data: Dict):
    """Salva status em arquivo JSON"""
    status_file = STATUS_DIR / f"{task_id}.json"
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2, ensure_ascii=False)

def load_status_file(task_id: str) -> Optional[Dict]:
    """Carrega status de arquivo JSON"""
    status_file = STATUS_DIR / f"{task_id}.json"
    if status_file.exists():
        try:
            # Verificar se o arquivo não está vazio
            if status_file.stat().st_size == 0:
                return None
            
            with open(status_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return None
                return json.loads(content)
        except json.JSONDecodeError as e:
            # Se o JSON estiver corrompido, tentar recriar do dicionário em memória
            if task_id in processing_tasks:
                # Salvar novamente para corrigir o arquivo
                update_status_file(task_id, processing_tasks[task_id])
                return processing_tasks[task_id]
            return None
        except Exception as e:
            # Qualquer outro erro, retornar None
            return None
    return None

@app.get("/api")
async def api_info():
    return {"message": "API de Fotogrametria - Sistema de Geração de Modelos 3D", "version": "1.0.0"}

@app.post("/api/process/start", response_model=ProcessingStatus)
async def start_processing(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """Inicia o processamento de geração do modelo 3D"""
    task_id = str(uuid.uuid4())
    
    # Criar registro da tarefa
    task_data = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Aguardando início...",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "error": None,
        "estimated_time_remaining": None,
        "current_step": "Aguardando",
        "logs": [],
        "input_path": request.input_path,
        "output_path": request.output_path,
        "metodo": request.metodo
    }
    
    processing_tasks[task_id] = task_data
    update_status_file(task_id, task_data)
    
    # Iniciar processamento em background
    thread = threading.Thread(target=run_pipeline_async, args=(task_id, request))
    thread.daemon = True
    thread.start()
    
    return ProcessingStatus(**task_data)

@app.get("/api/process/status/{task_id}", response_model=ProcessingStatus)
async def get_status(task_id: str):
    """Retorna o status de uma tarefa de processamento"""
    try:
        # Tentar carregar do arquivo primeiro
        status_data = load_status_file(task_id)
        
        if not status_data:
            # Verificar no dicionário em memória
            if task_id not in processing_tasks:
                raise HTTPException(status_code=404, detail="Tarefa não encontrada")
            status_data = processing_tasks[task_id]
        
        # Atualizar estimativa de tempo se estiver rodando
        if status_data.get("status") == "running" and status_data.get("start_time"):
            try:
                start_time = datetime.fromisoformat(status_data["start_time"])
                elapsed = (datetime.now() - start_time).total_seconds()
                progress = status_data.get("progress", 0)
                estimated = calculate_estimated_time(progress, elapsed)
                status_data["estimated_time_remaining"] = estimated
            except (ValueError, KeyError):
                # Se houver erro ao calcular tempo, continuar sem estimativa
                pass
        
        # Garantir que logs existem
        if "logs" not in status_data:
            status_data["logs"] = []
        
        # Garantir que todos os campos obrigatórios existem
        required_fields = ["task_id", "status", "progress", "message"]
        for field in required_fields:
            if field not in status_data:
                if field == "task_id":
                    status_data["task_id"] = task_id
                elif field == "status":
                    status_data["status"] = "unknown"
                elif field == "progress":
                    status_data["progress"] = 0.0
                elif field == "message":
                    status_data["message"] = "Status desconhecido"
        
        return ProcessingStatus(**status_data)
    except HTTPException:
        raise
    except Exception as e:
        # Se houver qualquer erro, tentar retornar dados básicos
        if task_id in processing_tasks:
            return ProcessingStatus(**processing_tasks[task_id])
        raise HTTPException(status_code=500, detail=f"Erro ao carregar status: {str(e)}")

@app.get("/api/process/{task_id}/logs")
async def get_logs(task_id: str, limit: int = 100):
    """Retorna os logs de uma tarefa"""
    status_data = load_status_file(task_id)
    
    if not status_data:
        if task_id not in processing_tasks:
            raise HTTPException(status_code=404, detail="Tarefa não encontrada")
        status_data = processing_tasks[task_id]
    
    logs = status_data.get("logs", [])
    
    # Retornar apenas os últimos N logs
    if limit > 0:
        logs = logs[-limit:]
    
    return {"task_id": task_id, "logs": logs, "total": len(status_data.get("logs", []))}

@app.get("/api/models/list")
async def list_models():
    """Lista todos os modelos 3D disponíveis"""
    output_path = Path("resources/output")
    models = []
    
    if output_path.exists():
        # Procurar por arquivos PLY
        for ply_file in output_path.rglob("*.ply"):
            models.append({
                "name": ply_file.name,
                "path": str(ply_file.relative_to(output_path)),
                "full_path": str(ply_file),
                "size": ply_file.stat().st_size,
                "modified": datetime.fromtimestamp(ply_file.stat().st_mtime).isoformat()
            })
    
    return {"models": models}

@app.get("/api/models/{model_path:path}")
async def get_model(model_path: str):
    """Serve um arquivo de modelo 3D"""
    output_path = Path("resources/output")
    model_file = output_path / model_path
    
    # Verificar segurança (não permitir acesso fora do diretório output)
    try:
        model_file.resolve().relative_to(output_path.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Acesso negado")
    
    if not model_file.exists():
        raise HTTPException(status_code=404, detail="Modelo não encontrado")
    
    return FileResponse(
        path=str(model_file),
        media_type="application/octet-stream",
        filename=model_file.name
    )

# Montar arquivos estáticos (frontend) - DEVE SER ANTES DE OUTRAS ROTAS
frontend_dir = Path("frontend")
assets_dir = Path("assets")
if frontend_dir.exists():
    # Servir arquivos estáticos do frontend
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
# Servir assets (logo, imagens, etc.)
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

# Rota raiz para servir o frontend (deve ser a última rota definida)
@app.get("/")
async def serve_frontend():
    index_file = frontend_dir / "index.html"
    if index_file.exists() and frontend_dir.exists():
        return FileResponse(str(index_file))
    return {"message": "Frontend não encontrado. Acesse /api para informações da API."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

