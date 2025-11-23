"""
Wrapper para DenseService que adiciona callbacks de progresso
"""
from pathlib import Path
from src.dense.reconstruir_dense import DenseService
from src.utils.log import Log
import time
import threading

class DenseServiceWrapper:
    def __init__(self, main_instance, output_path: str, task_id: str, start_time: float):
        self.main = main_instance
        self.output_path = output_path
        self.task_id = task_id
        self.start_time = start_time
        self.dense_service = None
        
    def iniciar_processamento(self):
        """Inicia o processamento com callbacks de progresso"""
        # Importar aqui para evitar importação circular
        import sys as sys_module
        from pathlib import Path
        sys_module.path.insert(0, str(Path(__file__).parent.parent))
        from api.main import update_task_status, add_log
        
        # Progresso estimado para cada etapa do COLMAP
        steps = [
            ("Verificando imagens...", 25.0),
            ("Extraindo características das imagens...", 30.0),
            ("Correspondendo imagens...", 40.0),
            ("Reconstruindo estrutura esparsa (SFM)...", 50.0),
            ("Corrigindo distorção das imagens...", 60.0),
            ("Gerando mapas de profundidade (Patch Match Stereo)...", 70.0),
        ]
        
        step_index = 0
        
        # Criar serviço denso
        self.dense_service = DenseService(
            base_path=self.output_path,
            max_image_size=2000,
            use_exhaustive_match=True
        )
        
        # Etapa 1: Gerar caminhos e verificar imagens
        step_name, progress = steps[step_index]
        update_task_status(self.task_id, step_name, progress, "COLMAP - Verificação", self.start_time)
        add_log(self.task_id, f"🔍 {step_name}")
        self.dense_service.gerar_caminhos()
        self.dense_service._verificar_imagens()
        step_index += 1
        
        # Etapa 2: Extrair características
        step_name, progress = steps[step_index]
        update_task_status(self.task_id, step_name, progress, "COLMAP - Extração", self.start_time)
        add_log(self.task_id, f"🔍 {step_name}")
        add_log(self.task_id, "⏳ Esta etapa pode levar vários minutos ou horas dependendo do número de imagens...")
        
        # Contar imagens para estimar progresso
        num_images = len(list(self.dense_service.images.glob("*.*")))
        add_log(self.task_id, f"📊 Processando {num_images} imagens...")
        
        # Criar callback para logs periódicos
        import threading
        import time as time_module
        
        last_update = time_module.time()
        processed_count = [0]  # Usar lista para permitir modificação em closure
        
        def progress_callback(line):
            nonlocal last_update
            current_time = time_module.time()
            
            # Atualizar a cada 30 segundos com mensagem de progresso
            if current_time - last_update > 30:
                # Tentar extrair informações da linha
                if "image" in line.lower() or "processing" in line.lower():
                    add_log(self.task_id, f"📸 {line[:100]}")  # Limitar tamanho da linha
                else:
                    from datetime import datetime
                    add_log(self.task_id, f"⏳ Processando... (última atualização: {datetime.now().strftime('%H:%M:%S')})")
                last_update = current_time
        
        # Iniciar thread para monitorar progresso
        def monitor_progress():
            while True:
                time_module.sleep(60)  # A cada minuto
                elapsed = time_module.time() - self.start_time
                elapsed_str = f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m"
                add_log(self.task_id, f"⏱️ Ainda processando... Tempo decorrido: {elapsed_str}")
        
        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()
        
        try:
            self.dense_service.extrair_caracteristicas(log_callback=progress_callback)
            add_log(self.task_id, "✅ Extração de características concluída!")
        except Exception as e:
            add_log(self.task_id, f"❌ Erro na extração: {str(e)}")
            raise
        finally:
            # O thread monitor será finalizado automaticamente quando a função terminar
            pass
        
        step_index += 1
        
        # Etapa 3: Match de imagens
        step_name, progress = steps[step_index]
        update_task_status(self.task_id, step_name, progress, "COLMAP - Correspondência", self.start_time)
        add_log(self.task_id, f"🔗 {step_name}")
        add_log(self.task_id, "⏳ Comparando todas as imagens...")
        
        def match_callback(line):
            if "matching" in line.lower() or "image" in line.lower() or "pair" in line.lower():
                add_log(self.task_id, f"🔗 {line[:100]}")
        
        self.dense_service.match_images(log_callback=match_callback)
        step_index += 1
        
        # Etapa 4: Reconstrução SFM
        step_name, progress = steps[step_index]
        update_task_status(self.task_id, step_name, progress, "COLMAP - SFM", self.start_time)
        add_log(self.task_id, f"🏗️ {step_name}")
        add_log(self.task_id, "⏳ Esta é uma das etapas mais demoradas...")
        self.dense_service.reconstruir_sfm()
        step_index += 1
        
        # Etapa 5: Undistort
        step_name, progress = steps[step_index]
        update_task_status(self.task_id, step_name, progress, "COLMAP - Correção", self.start_time)
        add_log(self.task_id, f"📐 {step_name}")
        self.dense_service.undistort()
        step_index += 1
        
        # Etapa 6: Patch Match Stereo (mais demorada)
        step_name, progress = steps[step_index]
        update_task_status(self.task_id, step_name, progress, "COLMAP - Patch Match", self.start_time)
        add_log(self.task_id, f"🌊 {step_name}")
        add_log(self.task_id, "⏳ Esta é a etapa mais demorada. Pode levar vários minutos...")
        self.dense_service.patchmatch_stereo()
        
        add_log(self.task_id, "✅ Reconstrução densa concluída com sucesso!")

