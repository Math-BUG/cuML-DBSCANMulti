#!/bin/bash
# Baixa os cabeçalhos/bibliotecas C++ de RAFT, RMM, cuVS e rapids_logger sem pip/venv.
#
# Por que existe: os wheels impõem uma versão mínima de Python para serem INSTALADOS, mas o
# build CUDA precisa dos cabeçalhos e das bibliotecas C++ de RMM/cuVS, não dos pacotes
# Python importáveis. Um wheel é um zip; extrair `include/` e `lib64/` não requer instalar
# nada. Assim o binário compila mesmo sem venv ou com Python antigo demais para o RAPIDS.
#
# Uso:
#   bash scripts/fetch_rapids_headers.sh
#   RAPIDS_VERSION=26.2.0 bash scripts/fetch_rapids_headers.sh
#
# Resultado (o Makefile encontra sozinho):
#   third_party/rapids_headers/libraft/include/raft/...
#   third_party/rapids_headers/librmm/include/rmm/...
#   third_party/rapids_headers/librmm/include/rapids/...   <- CCCL do RAPIDS
#   third_party/rapids_headers/{librmm,libcuvs,rapids_logger}/lib64/...

set -euo pipefail

# Precisa bater com a tag do cuML vendorizado em third_party/cuml (ver VENDORED.md).
RAPIDS_VERSION="${RAPIDS_VERSION:-26.2.0}"
CUDA_SUFFIX="${CUDA_SUFFIX:-cu12}"
ARCH_TAG="${ARCH_TAG:-x86_64}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$PROJECT_DIR/third_party/rapids_headers"

command -v curl >/dev/null 2>&1 || { echo "erro: curl não encontrado" >&2; exit 1; }
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "erro: python3 não encontrado" >&2; exit 1; }

echo "versão RAPIDS: $RAPIDS_VERSION ($CUDA_SUFFIX, $ARCH_TAG)"
echo "destino:       $DEST"

TMPDIR_LOCAL="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

mkdir -p "$DEST"

for pkg in libraft librmm libcuvs; do
  index="https://pypi.nvidia.com/${pkg}-${CUDA_SUFFIX}/"
  pattern="${pkg}_${CUDA_SUFFIX}-${RAPIDS_VERSION}-py3-none-manylinux[^\"#]*${ARCH_TAG}\.whl"

  echo "--- $pkg"
  file="$(curl -sL "$index" | grep -oE "$pattern" | head -1)"
  if [ -z "$file" ]; then
    echo "erro: não achei wheel de $pkg versão $RAPIDS_VERSION em $index" >&2
    echo "      versões disponíveis:" >&2
    curl -sL "$index" | grep -oE "${pkg}_${CUDA_SUFFIX}-[0-9.]+-" | sort -u | tail -8 >&2
    exit 1
  fi

  echo "    baixando $file"
  curl -sL --fail "${index}${file}" -o "$TMPDIR_LOCAL/${pkg}.whl"

  # zipfile da stdlib: funciona em qualquer Python 3, sem depender do unzip do sistema
  # (algumas versões do unzip não expandem '*' através de '/').
  # include/ para compilar e lib64/ para linkar: RMM não é header-only (librmm.so tem
  # rmm::cuda_stream_view, rmm::device_buffer, rmm::bad_alloc...).
  "$PYTHON_BIN" - "$TMPDIR_LOCAL/${pkg}.whl" "${pkg}/" "$DEST" <<'PY'
import sys, zipfile

whl, prefix, dest = sys.argv[1], sys.argv[2], sys.argv[3]
wanted = (prefix + "include/", prefix + "lib64/")
with zipfile.ZipFile(whl) as z:
    members = [m for m in z.namelist() if m.startswith(wanted) and not m.endswith("/")]
    if not members:
        raise SystemExit(f"erro: nenhum arquivo sob {wanted} em {whl}")
    z.extractall(dest, members)
    headers = sum(1 for m in members if "/include/" in m)
    print(f"    {headers} cabeçalhos + {len(members) - headers} arquivos de lib")
PY
done

# raft/core/logger.hpp inclui estes headers mesmo quando o shim local substitui o logger
# do cuML. Este pacote também fornece a biblioteca requerida pelo link.
echo "--- rapids_logger"
logger_index="https://pypi.org/simple/rapids-logger/"
logger_url="$(curl -sL "$logger_index" \
  | grep -oE "https://[^\"#]*rapids_logger-[0-9.]+-py3-none-manylinux[^\"#]*${ARCH_TAG}\.whl" \
  | tail -1)"
if [ -z "$logger_url" ]; then
  echo "erro: não achei wheel de rapids-logger em $logger_index" >&2
  exit 1
fi
echo "    baixando $(basename "$logger_url")"
curl -sL --fail "$logger_url" -o "$TMPDIR_LOCAL/rapids_logger.whl"
"$PYTHON_BIN" - "$TMPDIR_LOCAL/rapids_logger.whl" "rapids_logger/" "$DEST" <<'PY'
import sys, zipfile

whl, prefix, dest = sys.argv[1], sys.argv[2], sys.argv[3]
wanted = (prefix + "include/", prefix + "lib64/")
with zipfile.ZipFile(whl) as z:
    members = [m for m in z.namelist() if m.startswith(wanted) and not m.endswith("/")]
    if not members:
        raise SystemExit(f"erro: nenhum arquivo sob {wanted} em {whl}")
    z.extractall(dest, members)
    print(f"    {len(members)} arquivos extraídos (headers + biblioteca)")
PY

echo "--- verificação ---"
missing=0
for h in libcuvs/include/cuvs/distance/distance.hpp \
         libcuvs/include/cuvs/neighbors/epsilon_neighborhood.hpp \
         libraft/include/raft/core/device_mdspan.hpp \
         libraft/include/raft/core/handle.hpp \
         libraft/include/raft/core/error.hpp \
         libraft/include/raft/sparse/csr.hpp \
         libraft/include/raft/sparse/convert/csr.cuh \
         libraft/include/raft/label/classlabels.cuh \
         libraft/include/raft/label/merge_labels.cuh \
         libraft/include/raft/util/cuda_dev_essentials.cuh \
         libraft/include/raft/util/cuda_utils.cuh \
         libraft/include/raft/util/cudart_utils.hpp \
         librmm/include/rmm/cuda_stream_view.hpp \
         librmm/include/rmm/device_uvector.hpp \
         rapids_logger/include/rapids_logger/log_levels.h \
         rapids_logger/include/rapids_logger/logger.hpp; do
  if [ -f "$DEST/$h" ]; then
    echo "  ok      $h"
  else
    echo "  FALTA   $h"
    missing=1
  fi
done

check_library() {
  local label="$1" relative="$2"
  shift 2
  for candidate in "$@"; do
    if [ -f "$candidate" ]; then
      echo "  ok      $label: ${candidate#"$DEST/"}"
      return 0
    fi
  done
  echo "  FALTA   $label em $relative/lib64 (esperado .so ou .a)"
  return 1
}

check_library RMM librmm \
  "$DEST/librmm/lib64/librmm.so" "$DEST/librmm/lib64/librmm.a" || missing=1
check_library rapids_logger rapids_logger \
  "$DEST/rapids_logger/lib64/librapids_logger.so" \
  "$DEST/rapids_logger/lib64/librapids_logger.a" || missing=1
check_library cuVS libcuvs \
  "$DEST/libcuvs/lib64/libcuvs.so" "$DEST/libcuvs/lib64/libcuvs.a" || missing=1
if ! check_library RAFT libraft \
  "$DEST/libraft/lib64/libraft.so" "$DEST/libraft/lib64/libraft.a"; then
  echo "  aviso   libraft e opcional; use LINK_RAFT=0 se ela nao vier no wheel" >&2
fi

if [ -d "$DEST/librmm/include/rapids/cuda" ]; then
  echo "  ok      CCCL do RAPIDS em librmm/include/rapids (cuda/, thrust/, cub/)"
else
  echo "  aviso   CCCL do RAPIDS não encontrado; o build usará o do CUDA Toolkit" >&2
fi

[ "$missing" = "0" ] || exit 1

cat <<EOF

Pronto. Agora dá para compilar sem venv:

  make CUDA_ARCH=sm_80

O Makefile encontra esses cabeçalhos sozinho. Para o baseline cuML rode depois
scripts/setup_env.sh.
EOF
