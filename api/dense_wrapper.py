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
            
            # Logar todas as linhas importantes em tempo real
            # Limitar tamanho para evitar logs muito longos
            log_line = line[:200] if len(line) > 200 else line
            
            # Adicionar emoji baseado no conteúdo
            if "error" in line.lower() or "failed" in line.lower():
                emoji = "❌"
            elif "complete" in line.lower() or "finished" in line.lower() or "done" in line.lower():
                emoji = "✅"
            elif "processing" in line.lower() or "image" in line.lower():
                emoji = "📸"
            elif "progress" in line.lower() or "%" in line:
                emoji = "📊"
            elif "time" in line.lower() or "elapsed" in line.lower():
                emoji = "⏱️"
            elif "[Processando...]" in line or "[Heartbeat]" in line:
                emoji = "💓"
            else:
                emoji = "🔍"
            
            add_log(self.task_id, f"{emoji} {log_line}")
            last_update = current_time
        
        # Variável para rastrear última atividade
        last_activity = [time_module.time()]
        
        def update_activity():
            last_activity[0] = time_module.time()
        
        # Modificar callback para atualizar atividade
        original_callback = progress_callback
        def activity_callback(line):
            update_activity()
            original_callback(line)
        
        # Iniciar thread para monitorar progresso e detectar travamentos
        last_warning_time = [0]  # Rastrear quando foi dado o último aviso
        
        def monitor_progress():
            while True:
                time_module.sleep(120)  # Verificar a cada 2 minutos (menos frequente)
                elapsed = time_module.time() - self.start_time
                elapsed_str = f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m"
                
                # Verificar se há atividade recente (últimos 10 minutos)
                time_since_activity = time_module.time() - last_activity[0]
                time_since_last_warning = time_module.time() - last_warning_time[0]
                
                # Só avisar se não há atividade há mais de 10 minutos E não avisamos nos últimos 5 minutos
                if time_since_activity > 600 and time_since_last_warning > 300:  # 10 min sem atividade, 5 min desde último aviso
                    minutes_stuck = int(time_since_activity // 60)
                    add_log(self.task_id, f"⚠️ AVISO: Nenhuma atividade detectada nos últimos {minutes_stuck} minutos. Processamento pode estar travado ou apenas demorando muito.")
                    last_warning_time[0] = time_module.time()
                elif time_since_activity <= 600:
                    # Há atividade recente, mostrar status positivo a cada 5 minutos
                    if elapsed % 300 < 120:  # A cada ~5 minutos
                        add_log(self.task_id, f"⏱️ Processando... Tempo decorrido: {elapsed_str} | Última atividade: {int(time_since_activity)}s atrás")
        
        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()
        
        # Usar callback com rastreamento de atividade
        progress_callback = activity_callback
        
        try:
            self.dense_service.extrair_caracteristicas(log_callback=activity_callback)
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
            # Logar todas as linhas importantes do matching
            log_line = line[:200] if len(line) > 200 else line
            if any(keyword in line.lower() for keyword in ["matching", "image", "pair", "match", "progress", "complete", "error"]):
                emoji = "🔗" if "matching" in line.lower() else "📊" if "progress" in line.lower() else "❌" if "error" in line.lower() else "✅" if "complete" in line.lower() else "🔍"
                add_log(self.task_id, f"{emoji} {log_line}")
        
        self.dense_service.match_images(log_callback=match_callback)
        step_index += 1
        
        # Função auxiliar para criar callbacks padronizados
        def create_log_callback(emoji_default, keywords, step_name):
            def callback(line):
                log_line = line[:200] if len(line) > 200 else line
                if any(keyword in line.lower() for keyword in keywords):
                    emoji = emoji_default
                    if "error" in line.lower() or "failed" in line.lower():
                        emoji = "❌"
                    elif "complete" in line.lower() or "finished" in line.lower():
                        emoji = "✅"
                    elif "progress" in line.lower() or "%" in line:
                        emoji = "📊"
                    add_log(self.task_id, f"{emoji} {log_line}")
            return callback
        
        # Função para executar etapa com callback
        def run_step_with_callback(step_func, callback, step_name):
            original_run = self.dense_service.run
            def run_with_callback(cmd, log_callback=None):
                def combined_callback(line):
                    callback(line)
                    if log_callback:
                        log_callback(line)
                return original_run(cmd, log_callback=combined_callback)
            self.dense_service.run = run_with_callback
            try:
                step_func()
            finally:
                self.dense_service.run = original_run
        
        # Etapa 4: Reconstrução SFM
        step_name, progress = steps[step_index]
        update_task_status(self.task_id, step_name, progress, "COLMAP - SFM", self.start_time)
        add_log(self.task_id, f"🏗️ {step_name}")
        add_log(self.task_id, "⏳ Esta é uma das etapas mais demoradas...")
        sfm_callback = create_log_callback("🏗️", ["reconstruction", "camera", "point", "image", "progress", "complete", "error", "iteration"], "SFM")
        run_step_with_callback(self.dense_service.reconstruir_sfm, sfm_callback, "SFM")
        step_index += 1
        
        # Etapa 5: Undistort
        step_name, progress = steps[step_index]
        update_task_status(self.task_id, step_name, progress, "COLMAP - Correção", self.start_time)
        add_log(self.task_id, f"📐 {step_name}")
        undistort_callback = create_log_callback("📐", ["undistort", "image", "processing", "complete", "error"], "Undistort")
        run_step_with_callback(self.dense_service.undistort, undistort_callback, "Undistort")
        step_index += 1
        
        # Etapa 6: Patch Match Stereo (mais demorada)
        step_name, progress = steps[step_index]
        update_task_status(self.task_id, step_name, progress, "COLMAP - Patch Match", self.start_time)
        add_log(self.task_id, f"🌊 {step_name}")
        add_log(self.task_id, "⏳ Esta é a etapa mais demorada. Pode levar vários minutos...")
        patchmatch_callback = create_log_callback("🌊", ["patch", "stereo", "depth", "processing", "progress", "complete", "error", "iteration", "%"], "Patch Match")
        run_step_with_callback(self.dense_service.patchmatch_stereo, patchmatch_callback, "Patch Match")
        
        add_log(self.task_id, "✅ Reconstrução densa concluída com sucesso!")

