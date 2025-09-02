# Fotogrametria Aplicada

Projeto para execução de um pipeline de fotogrametria e experimentos exploratórios via notebooks, incluindo geração de artefatos como nuvens de pontos, malhas e visualizações auxiliares.

- Repositório: este diretório
- Linguagem: Python
- Container: Docker (opcional)

## Tabela de Conteúdos
- [Visão Geral](#visão-geral)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Execução do Pipeline](#execução-do-pipeline)
- [Uso com Docker](#uso-com-docker)
- [Notebooks e Geração de Imagens](#notebooks-e-geração-de-imagens)
- [Testes](#testes)
- [Saída e Organização de Resultados](#saída-e-organização-de-resultados)
- [Dicas e Solução de Problemas](#dicas-e-solução-de-problemas)
- [Contribuição](#contribuição)
- [Licença](#licença)
- [Citação](#citação)

## Visão Geral
Este projeto reúne:
- Um pipeline executável para processar conjuntos de imagens e produzir artefatos fotogramétricos (ex.: reconstrução 3D).
- Notebooks para análise, validação e visualização dos resultados.
- Ambiente reprodutível via Docker (opcional) e instalação local via `requirements.txt`.

## Estrutura do Repositório
- `src/`: código-fonte do pipeline e utilitários.
- `notebooks/`: cadernos Jupyter com análises e visualizações.
- `resources/`: dados auxiliares e onde você pode armazenar imagens geradas pelos notebooks (ex.: `resources/figures/`).
- `tests/`: testes automatizados.
- `run_pipeline.py`: ponto de entrada do pipeline via linha de comando.
- `requirements.txt`: dependências em Python.
- `Dockerfile`: imagem para execução reprodutível.
- `setup.py`: instalação como pacote (opcional).
- `.vscode/`, `.idea/`, `.venv/`: configurações e ambiente de desenvolvimento (locais).

## Pré-requisitos
- Python 3.10+ (recomendado)
- Pip e virtualenv (ou equivalente)
- Opcional: Docker 24+ e Docker Compose

## Instalação
Crie e ative um ambiente virtual, depois instale as dependências:
```
bash
# Linux/macOS
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

```
powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
Se desejar instalar como pacote editável (opcional):
```
bash
pip install -e .
```
## Execução do Pipeline
Execute o script principal para ver as opções disponíveis:
```
bash
python run_pipeline.py --help
```
Exemplo de execução (ajuste caminhos e parâmetros conforme sua necessidade):
```
bash
python run_pipeline.py \
--input <CAMINHO_PARA_IMAGENS_DE_ENTRADA> \
--output ./resources/outputs \
--config <CAMINHO_PARA_ARQUIVO_DE_CONFIG_OPCIONAL>
```
Observações:
- Garanta que o diretório de saída exista ou que o pipeline possa criá-lo.

## Uso com Docker
Construa a imagem e rode o pipeline dentro do container:
```
bash
# Construir a imagem
docker build -t fotogrametria:latest .

# Executar o pipeline (montando o diretório do projeto no container)
docker run --rm -it -v "$PWD":/app -w /app fotogrametria:latest \
python run_pipeline.py --help
```
Exemplo com entrada e saída montadas:
```
bash
docker run --rm -it \
-v "$PWD":/app -w /app \
-v "<CAMINHO_LOCAL_IMAGENS>":/data/input \
-v "$PWD/resources/outputs":/data/output \
fotogrametria:latest \
python run_pipeline.py --input /data/input --output /data/output
```
## Notebooks e Geração de Imagens
Inicie o Jupyter Lab/Notebook e execute os cadernos do diretório `notebooks/`:
```
bash
jupyter lab
# ou
jupyter notebook
```
Recomendações:
- Execute as células em ordem para reproduzir análises.
- Salve figuras e artefatos no diretório `resources/figures/` (crie-o, se necessário) para versionamento e referência no README.

Exemplo (dentro de um notebook) para salvar uma figura com Matplotlib:
```
python
# Exemplo genérico de salvamento de figura
import os
import matplotlib.pyplot as plt

figures_dir = "resources/figures"
os.makedirs(figures_dir, exist_ok=True)

plt.figure(figsize=(6, 4))
plt.plot([0, 1, 2], [0, 1, 0])
plt.title("Curva de Exemplo")
plt.savefig(os.path.join(figures_dir, "curva_exemplo.png"), dpi=150, bbox_inches="tight")
plt.close()
```
Ao final, referencie as imagens aqui no README. Exemplos de resultados esperados:

- Visualização da nuvem de pontos:
  ![Nuvem de Pontos (exemplo)](resources/figures/nuvem_pontos_exemplo.png)

- Malha reconstruída:
  ![Malha Reconstruída (exemplo)](resources/figures/malha_reconstruida_exemplo.png)

- Correspondência de features entre imagens:
  ![Matches de Features (exemplo)](resources/figures/matches_features_exemplo.png)

Substitua os arquivos acima pelos gerados nos seus notebooks.

## Testes
Execute a suíte de testes:
```
bash
pytest -q
```
Ou para um módulo/arquivo específico:
```
bash
pytest -q tests/<ARQUIVO_OU_DIRETÓRIO>
```
## Saída e Organização de Resultados
Sugestão de organização:
- `resources/outputs/`: artefatos do pipeline (ex.: modelos, nuvens, relatórios).
- `resources/figures/`: imagens e gráficos exportados dos notebooks.
- `resources/tmp/`: intermediários temporários (se aplicável).

Inclua no `.gitignore` diretórios volumosos e gerados automaticamente, se ainda não estiverem.

## Dicas e Solução de Problemas
- Dependências nativas: alguns pacotes de visão computacional exigem bibliotecas do sistema (ex.: compilers, drivers). Instale-as previamente conforme seu SO.
- Memória e tempo: reconstruções 3D podem ser custosas. Comece com um subconjunto pequeno de imagens.
- Reprodutibilidade: fixe versões em `requirements.txt` e utilize Docker quando possível.
- CLI: use `--help` para inspecionar parâmetros e valores padrão.

## Contribuição
- Abra uma issue para discutir novas funcionalidades ou problemas.
- Faça fork, crie uma branch e envie um PR com descrição clara, passos de reprodução e, se possível, testes.

## Citação
Se você utilizar este projeto em trabalhos acadêmicos ou relatórios, por favor cite-o. Exemplo:
> Fotogrametria aplicada: Convertendo imagens 2D em modelos 3D. Fotogrametria Aplicada: pipeline e notebooks de análise. 2025. https://github.com/prandini-kaio/fotogrametria-aplicada.

