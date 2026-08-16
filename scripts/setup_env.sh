#!/bin/bash
# Cria o venv do projeto no ClusterGPU/UFV.
#
# Segue o mesmo padrão do setup_venv_cluster.sh do INF-494, que já funciona nesse
# cluster: carrega os módulos aqui dentro, isola o venv do site-packages do usuário e
# instala em etapas, verificando cada uma.
#
# Rode no NÓ DE LOGIN — é só download e instalação, não precisa de GPU nem de fila:
#
#   bash scripts/setup_env.sh
#   source .venv/bin/activate
#
# Instalar o cuML (baseline de comparação) é opcional e demorado; ative com:
#
#   INSTALL_CUML=1 bash scripts/setup_env.sh
#
# Sem o cuML dá para gerar datasets, compilar o binário CUDA e rodar o selftest. O que
# fica de fora é só a comparação de tempo contra o baseline.
#
# O venv fica em .venv. Se a HOME tiver cota apertada (o RAPIDS passa de 3 GB):
#
#   VENV_DIR=~/dados/venvs/dbscanmulti bash scripts/setup_env.sh

set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_CUML="${INSTALL_CUML:-0}"
USE_LOCK="${USE_LOCK:-0}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# O /tmp dos nós do cluster é compartilhado e pode estar cheio. Downloads, wheels e
# arquivos intermediários do pip ficam na partição de dados por padrão.
DBM_SETUP_STORAGE="${DBM_SETUP_STORAGE:-$HOME/dados/dbscanmulti/setup}"
export TMPDIR="${DBM_SETUP_TMPDIR:-$DBM_SETUP_STORAGE/tmp}"
export PIP_CACHE_DIR="${DBM_PIP_CACHE_DIR:-$DBM_SETUP_STORAGE/pip-cache}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# `criar_venv` remove o diretório antes de tentar novamente. Resolva e valide uma vez,
# antes de qualquer rm, para que VENV_DIR vazio, '/', '.', '..' ou a raiz do projeto nunca
# possa apagar uma árvore ampla por engano.
VENV_DIR="$(realpath -m -- "$VENV_DIR")"
case "$VENV_DIR" in
  ""|/|"$PROJECT_DIR"|"$(dirname "$PROJECT_DIR")")
    echo "erro: VENV_DIR inseguro: '$VENV_DIR'" >&2
    exit 2
    ;;
esac
if [ "$(dirname "$VENV_DIR")" = "$VENV_DIR" ]; then
  echo "erro: VENV_DIR resolve para uma raiz de sistema: '$VENV_DIR'" >&2
  exit 2
fi
case "$PROJECT_DIR/" in
  "$VENV_DIR/"*)
    echo "erro: VENV_DIR '$VENV_DIR' é ancestral do repositório '$PROJECT_DIR'" >&2
    exit 2
    ;;
esac

# Impede que o venv enxergue ~/.local/lib/pythonX.Y/site-packages. Sem isso, pacotes
# antigos da HOME (torch, numpy desatualizado) vazam para dentro do venv e quebram a
# instalação do RAPIDS de formas difíceis de diagnosticar.
export PYTHONNOUSERSITE=1

# PYTHONHOME/PYTHONPATH herdados do shell ou postos por módulos fazem o interpretador do
# venv recém-criado apontar para o lugar errado; o sintoma é o venv "sumir" na hora de
# rodar o ensurepip.
unset PYTHONHOME PYTHONPATH

echo "projeto: $PROJECT_DIR"
echo "venv:    $VENV_DIR"
echo "tmp:     $TMPDIR"
echo "pip cache: $PIP_CACHE_DIR"
echo "python:  $(command -v python3) ($(python3 --version 2>&1))"

# --- criação do venv ---------------------------------------------------------
# `python3 -m venv` falha de várias formas em cluster: symlink negado no sistema de
# arquivos da HOME, ensurepip quebrado, cota estourada, ou um venv anterior interrompido
# pela metade. O sintoma costuma ser sempre o mesmo e pouco informativo:
#   Error: [Errno 2] No such file or directory: '.../.venv/bin/python3'
# Por isso, quando a recriação é necessária, remover de forma controlada e tentar as
# variantes em ordem. Um venv sadio só é recriado com RECREATE_VENV=1.
VENV_CRIADO_NESTA_EXECUCAO=0
criar_venv() {
  if [ -e "$VENV_DIR" ] && [ ! -f "$VENV_DIR/pyvenv.cfg" ] &&
     [ "$VENV_CRIADO_NESTA_EXECUCAO" != 1 ]; then
    echo "erro: recuso remover '$VENV_DIR': diretório existente não parece um virtualenv" >&2
    return 2
  fi
  rm -rf -- "$VENV_DIR"
  VENV_CRIADO_NESTA_EXECUCAO=1
  python3 -m venv "$@" "$VENV_DIR" >/dev/null 2>&1 && [ -x "$VENV_DIR/bin/python" ]
}

if [ "${RECREATE_VENV:-0}" != "1" ] && [ -x "$VENV_DIR/bin/python" ] \
   && "$VENV_DIR/bin/python" -c 'import sys' 2>/dev/null; then
  # Reaproveita um venv sadio: rodar de novo só para acrescentar o RAPIDS não deve
  # jogar fora o que já está instalado. Use RECREATE_VENV=1 para forçar do zero.
  echo "venv existente reaproveitado ($VENV_DIR)"
elif criar_venv; then
  echo "venv criado"
elif criar_venv --copies; then
  echo "venv criado com --copies (symlinks não funcionaram neste sistema de arquivos)"
elif criar_venv --copies --without-pip; then
  echo "venv criado sem pip (ensurepip indisponível); instalando pip via get-pip.py"
  GET_PIP="$(mktemp)"
  trap 'rm -f "$GET_PIP"' EXIT
  curl -sSL --fail https://bootstrap.pypa.io/pip/get-pip.py -o "$GET_PIP" \
    || curl -sSL --fail https://bootstrap.pypa.io/get-pip.py -o "$GET_PIP"
  "$VENV_DIR/bin/python" "$GET_PIP"
else
  cat >&2 <<'EOF'

erro: não consegui criar o venv de nenhuma forma. Diagnóstico abaixo.
EOF
  echo "--- interpretador ---" >&2
  command -v python3 >&2 || echo "python3 não está no PATH" >&2
  python3 -c "import sys; print('executable:', sys.executable); print('prefix:', sys.prefix)" >&2 || true
  python3 -c "import ensurepip; print('ensurepip: ok')" >&2 \
    || echo "ensurepip: AUSENTE (falta o pacote python3-venv na imagem do nó)" >&2
  echo "--- espaço e cota ---" >&2
  df -h "$PROJECT_DIR" 2>/dev/null | tail -2 >&2 || true
  quota -s 2>/dev/null | tail -3 >&2 || echo "quota: comando indisponível" >&2
  echo "--- permissões ---" >&2
  ls -ld "$PROJECT_DIR" >&2 || true
  cat >&2 <<EOF

Saídas possíveis, conforme o diagnóstico acima:
  - cota/espaço estourado na HOME:  VENV_DIR=~/dados/venvs/dbscanmulti bash scripts/setup_env.sh
  - ensurepip ausente:              peça python3-venv ao suporte, ou use um módulo de Python
  - erro completo do venv:          python3 -m venv $VENV_DIR    (rode à mão para ver a mensagem)
EOF
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# Módulos só depois do venv pronto: eles existem para o nvcc, e o ambiente que carregam
# (PYTHONHOME, PYTHONPATH, LD_LIBRARY_PATH) já atrapalhou a criação do venv aqui.
module --force purge
module load GCCcore/12.2.0
module load CUDA/12.6.0
unset PYTHONHOME PYTHONPATH

if [ "$VENV_CRIADO_NESTA_EXECUCAO" = 1 ]; then
  python -m pip install --upgrade pip setuptools wheel
else
  # Num venv existente, instalar só o que faltar. `--upgrade` baixava uma nova versão de
  # setuptools a cada execução, antes mesmo de chegar à dependência realmente ausente.
  python -m pip install pip setuptools wheel
fi

if [ "$USE_LOCK" = "1" ]; then
  if [ ! -s requirements.lock.txt ]; then
    echo "erro: USE_LOCK=1 exige requirements.lock.txt versionado e não vazio" >&2
    exit 2
  fi
  echo "--- instalando ambiente congelado por requirements.lock.txt ---"
  python -m pip install --extra-index-url https://pypi.nvidia.com -r requirements.lock.txt
else
  echo "--- etapa 1/2: ferramentas (datasets, análise, testes) ---"
  python -m pip install -r requirements.txt
  python -c "import numpy, pytest, sklearn; print('deps basicas e testes ok')"

  if [ "$INSTALL_CUML" = "1" ]; then
    echo "--- etapa 2/2: RAPIDS (baseline cuML) — download grande, alguns minutos ---"
    python -m pip install -r requirements-cuml.txt
    python -c "import cupy, rapids_logger; from cuml.cluster import DBSCAN; print('cuml e rapids_logger ok')"
  else
    echo "--- etapa 2/2: RAPIDS pulado (use INSTALL_CUML=1 para instalar o baseline) ---"
  fi
fi

if [ "$USE_LOCK" = "1" ]; then
  python -m pip freeze --all | diff -u requirements.lock.txt -
  echo "ambiente confere exatamente com requirements.lock.txt"
else
  python -m pip freeze --all > "$PROJECT_DIR/requirements.lock.txt"
  echo "versões exatas registradas em requirements.lock.txt; revise e versione o arquivo"
fi

# --- cabeçalhos de RAFT/RMM para compilar o binário --------------------------
# Vêm dos wheels lib* quando o RAPIDS está instalado; senão, extraídos direto dos
# wheels (um wheel é um zip), sem pip.
if python -c "import libraft" 2>/dev/null; then
  python -c "import libraft, os; print('RAFT_INCLUDE =', os.path.join(os.path.dirname(libraft.__file__), 'include'))"
  python -c "import librmm, os; print('RMM_INCLUDE  =', os.path.join(os.path.dirname(librmm.__file__), 'include'))"
elif [ ! -f "third_party/rapids_headers/libraft/include/raft/core/handle.hpp" ]; then
  echo "--- baixando cabeçalhos de RAFT/RMM ---"
  bash scripts/fetch_rapids_headers.sh
fi

make check-headers

cat <<EOF

Pronto. Próximos passos:

  source $VENV_DIR/bin/activate
  sbatch scripts/check_gpu.slm     # compila e roda o selftest em um job curto
  squeue --me

EOF
