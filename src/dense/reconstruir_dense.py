import os
import platform
import subprocess
import time
import shutil
from pathlib import Path

from src.utils.log import Log
from src.config.settings import Settings


class DenseService:

    def __init__(self, base_path: str, max_image_size: int, use_exhaustive_match: bool):
        self.base_path = Path(base_path)
        self.max_image_size = max_image_size
        self.use_exhaustive_match = use_exhaustive_match

    def gerar_caminhos(self):
        self.images = self.base_path / "images"
        self.sparse = self.base_path / "sparse"
        self.dense = self.base_path / "dense"
        self.database = self.base_path / "database"

        self.sparse.mkdir(parents=True, exist_ok=True)
        self.dense.mkdir(parents=True, exist_ok=True)

    def iniciar_processamento(self):
        self.gerar_caminhos()
        self._verificar_imagens()
        self.extrair_caracteristicas()
        self.match_images()
        self.reconstruir_sfm()
        self.undistort()
        self.patchmatch_stereo()
        Log.success("[COLMAP] Pipeline finalizado com sucesso!")

    def extrair_caracteristicas(self, log_callback=None):
        self.run([
            "colmap", "feature_extractor",
            "--database_path", str(self.database),
            "--image_path", str(self.images),
            "--ImageReader.single_camera", "1"
        ], log_callback=log_callback)

    def match_images(self, log_callback=None):
        matcher = "exhaustive_matcher" if self.use_exhaustive_match else "sequential_matcher"
        self.run([
            "colmap", matcher,
            "--database_path", str(self.database)
        ], log_callback=log_callback)

    def reconstruir_sfm(self):
        self.run([
            "colmap", "mapper",
            "--database_path", str(self.database),
            "--image_path", str(self.images),
            "--output_path", str(self.sparse)
        ])

    def undistort(self):
        self.run([
            "colmap", "image_undistorter",
            "--image_path", str(self.images),
            "--input_path", str(self.sparse / "0"),
            "--output_path", str(self.dense),
            "--output_type", "COLMAP",
            "--max_image_size", str(self.max_image_size)
        ])

    def patchmatch_stereo(self):
        self.run([
            "colmap", "patch_match_stereo",
            "--workspace_path", str(self.dense),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true"
        ])

    @staticmethod
    def find_colmap_executable():
        """Encontra o executável do COLMAP"""
        # 1. Verificar variável de ambiente primeiro (mais confiável)
        colmap_env = os.environ.get("COLMAP_PATH") or os.environ.get("COLMAP_EXE_PATH")
        if colmap_env:
            colmap_env = colmap_env.strip().strip('"').strip("'")
            if Path(colmap_env).exists():
                Log.info(f"[COLMAP] Encontrado via variável de ambiente: {colmap_env}")
                return colmap_env
            else:
                Log.warning(f"[COLMAP] Caminho da variável de ambiente não existe: {colmap_env}")
        
        # 2. Tentar carregar Settings (pode falhar se pydantic não estiver disponível)
        try:
            settings = Settings()
            if settings.COLMAP_PATH:
                colmap_path = settings.COLMAP_PATH.strip().strip('"').strip("'")
                if Path(colmap_path).exists():
                    Log.info(f"[COLMAP] Encontrado via Settings: {colmap_path}")
                    return colmap_path
                else:
                    Log.warning(f"[COLMAP] Caminho do Settings não existe: {colmap_path}")
        except Exception as e:
            Log.warning(f"[COLMAP] Erro ao carregar Settings: {e}")
        
        # 3. Verificar se está no PATH (tentar com .exe no Windows)
        if platform.system() == "Windows":
            # Tentar colmap.exe primeiro
            colmap_path = shutil.which("colmap.exe")
            if colmap_path:
                Log.info(f"[COLMAP] Encontrado no PATH: {colmap_path}")
                return colmap_path
            # Tentar apenas colmap
            colmap_path = shutil.which("colmap")
            if colmap_path:
                Log.info(f"[COLMAP] Encontrado no PATH: {colmap_path}")
                return colmap_path
        else:
            colmap_path = shutil.which("colmap")
            if colmap_path:
                Log.info(f"[COLMAP] Encontrado no PATH: {colmap_path}")
                return colmap_path
        
        # 4. Procurar em locais comuns no Windows
        if platform.system() == "Windows":
            common_paths = [
                r"C:\COLMAP\colmap.exe",
                r"C:\Program Files\COLMAP\colmap.exe",
                r"C:\Program Files (x86)\COLMAP\colmap.exe",
                r"D:\COLMAP\colmap.exe",
                os.path.expanduser(r"~\COLMAP\colmap.exe"),
                os.path.expanduser(r"~\AppData\Local\COLMAP\colmap.exe"),
            ]
            for path in common_paths:
                if Path(path).exists():
                    Log.info(f"[COLMAP] Encontrado em local padrão: {path}")
                    return path
        
        # 5. Procurar em locais comuns no Linux/Mac
        else:
            common_paths = [
                "/usr/local/bin/colmap",
                "/usr/bin/colmap",
                os.path.expanduser("~/colmap/build/src/exe/colmap"),
            ]
            for path in common_paths:
                if Path(path).exists():
                    Log.info(f"[COLMAP] Encontrado em local padrão: {path}")
                    return path
        
        Log.error("[COLMAP] Não foi possível encontrar o executável do COLMAP")
        return None

    @staticmethod
    def run(cmd, log_callback=None):
        # Substituir "colmap" pelo caminho completo se necessário
        if cmd[0] == "colmap" or cmd[0].endswith("colmap") or cmd[0].endswith("colmap.exe"):
            colmap_exe = DenseService.find_colmap_executable()
            if colmap_exe:
                # Normalizar caminho para Windows
                colmap_exe = str(Path(colmap_exe).resolve())
                cmd[0] = colmap_exe
                Log.info(f"[COLMAP] Usando executável: {colmap_exe}")
            else:
                error_msg = (
                    "COLMAP não encontrado!\n\n"
                    "Opções para resolver:\n"
                    "1. Configure a variável de ambiente COLMAP_PATH:\n"
                    "   setx COLMAP_PATH \"C:\\caminho\\para\\colmap.exe\"\n"
                    "2. Crie um arquivo .env na raiz do projeto com:\n"
                    "   COLMAP_PATH=C:\\caminho\\para\\colmap.exe\n"
                    "3. Adicione colmap.exe ao PATH do Windows\n"
                    "4. Instale o COLMAP em um dos locais padrão:\n"
                    "   - C:\\COLMAP\\colmap.exe\n"
                    "   - C:\\Program Files\\COLMAP\\colmap.exe\n\n"
                    "Verifique se o caminho está correto e se o arquivo existe."
                )
                Log.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        Log.info(f'[COLMAP] Executando: {" ".join(cmd)}')

        full_env = os.environ.copy()
        system = platform.system()

        if system == "Linux":
            full_env['CUDA_PATH'] = '/usr/local/cuda'
            full_env['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + full_env.get('LD_LIBRARY_PATH', '')
            full_env["QT_QPA_PLATFORM"] = "offscreen"

            # # Adicionar diretório do COLMAP ao PATH para encontrar DLLs
            # if cmd[0].endswith(".exe") or "colmap" in cmd[0].lower():
            #     colmap_exe_path = Path(cmd[0])
            #     if colmap_exe_path.exists():
            #         colmap_dir = str(colmap_exe_path.parent)
            #         # Adicionar diretório do COLMAP ao início do PATH
            #         full_env["PATH"] = colmap_dir + os.pathsep + full_env.get("PATH", "")
            #         Log.info(f"[COLMAP] Adicionando ao PATH: {colmap_dir}")
            #
            #         # Também verificar se há subdiretório lib/ e adicionar
            #         colmap_lib = colmap_exe_path.parent / "lib"
            #         if colmap_lib.exists():
            #             full_env["PATH"] = str(colmap_lib) + os.pathsep + full_env["PATH"]
            #             Log.info(f"[COLMAP] Adicionando lib ao PATH: {colmap_lib}")
            #
            # # Se CUDA estiver instalado no caminho padrão, adicione ao PATH
            # cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
            # if Path(cuda_path).exists():
            #     full_env["PATH"] = cuda_path + os.pathsep + full_env.get("PATH", "")
            #
            # # Tentar outras versões comuns do CUDA
            # for cuda_version in ["v12.1", "v12.0", "v11.8", "v11.7"]:
            #     cuda_path_alt = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{cuda_version}\bin"
            #     if Path(cuda_path_alt).exists():
            #         full_env["PATH"] = cuda_path_alt + os.pathsep + full_env.get("PATH", "")
            #         Log.info(f"[COLMAP] Adicionando CUDA ao PATH: {cuda_path_alt}")
            #         break

        try:
            # Executar com captura de saída em tempo real
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=full_env,
                universal_newlines=True,
                bufsize=1
            )
            
            # Ler saída linha por linha
            last_log_time = time.time()
            line_count = 0
            last_heartbeat = time.time()
            
            # Palavras-chave importantes que sempre devem ser logadas
            important_keywords = [
                'error', 'warning', 'failed', 'exception', 'traceback',
                'processing', 'image', 'camera', 'point', 'match', 'reconstruction',
                'progress', 'complete', 'finished', 'done', 'elapsed', 'time',
                'percent', '%', 'remaining', 'estimated',
                'registering', 'incremental', 'pipeline', 'sees', 'points', 'frames',
                'colmap', 'feature', 'extractor', 'mapper', 'stereo', 'patch'
            ]
            
            for line in process.stdout:
                if line:
                    line = line.strip()
                    if line:
                        current_time = time.time()
                        should_log = False
                        
                        # Sempre logar linhas importantes
                        line_lower = line.lower()
                        is_important = any(keyword in line_lower for keyword in important_keywords)
                        
                        if is_important:
                            should_log = True
                        # Logar a cada 3 linhas (mais frequente para capturar mais atividade)
                        elif line_count % 3 == 0:
                            should_log = True
                        # Logar a cada 10 segundos (heartbeat mais frequente)
                        elif (current_time - last_log_time) > 10:
                            should_log = True
                            # Adicionar indicador de que está processando
                            line = f"[Processando...] {line}"
                        
                        # IMPORTANTE: Sempre atualizar heartbeat se há qualquer linha (indica atividade)
                        if is_important or line_count % 10 == 0:
                            last_heartbeat = current_time
                        
                        if should_log and log_callback:
                            log_callback(line)
                            last_log_time = current_time
                        
                        line_count += 1
                        
                        # Heartbeat: se passou mais de 1 minuto sem log importante, enviar heartbeat
                        if (current_time - last_heartbeat) > 60:
                            if log_callback:
                                log_callback(f"[Heartbeat] Processamento ainda em andamento... (linha {line_count})")
                            last_heartbeat = current_time
            
            # Aguardar conclusão
            return_code = process.wait()
            
            if return_code != 0:
                error_msg = None
                # Melhorar mensagem de erro para problemas comuns no Windows
                if system == "Windows" and return_code == 3221225786:  # 0xC0000135 - STATUS_DLL_NOT_FOUND
                    error_msg = (
                        f"COLMAP falhou com erro de DLL não encontrada (código: {return_code}).\n\n"
                        f"Possíveis soluções:\n"
                        f"1. Instale o Visual C++ Redistributable mais recente:\n"
                        f"   https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                        f"2. Verifique se todas as DLLs do COLMAP estão no mesmo diretório que colmap.exe\n"
                        f"3. Verifique se o diretório do COLMAP está no PATH do sistema\n"
                        f"4. Se estiver usando GPU, verifique se os drivers CUDA estão instalados corretamente\n"
                        f"5. Tente executar colmap.exe manualmente para ver mensagens de erro mais detalhadas\n\n"
                        f"Comando executado: {' '.join(cmd)}"
                    )
                    Log.error(error_msg)
                raise subprocess.CalledProcessError(return_code, cmd, output=error_msg)
                
        except FileNotFoundError:
            Log.error("COLMAP não encontrado. Adicione colmap.exe ao PATH do Windows "
                      "ou configure o caminho absoluto no código.")
            raise

    def _verificar_imagens(self):
        """Verifica se existem imagens no diretório"""
        imagens = list(self.images.glob("*.*"))
        formatos_suportados = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        imagens_validas = [img for img in imagens if img.suffix.lower() in formatos_suportados]

        if not imagens_validas:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em {self.images}")

        Log.success(f"[COLMAP] Encontradas {len(imagens_validas)} imagens")
