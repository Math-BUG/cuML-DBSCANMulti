#!/bin/bash
# Emite somente o orçamento em MB no stdout; diagnósticos vão para stderr.

set -euo pipefail

native_auto="${NATIVE_AUTO_MEMORY:-0}"
requested="${MAX_MBYTES_PER_BATCH:-}"
percent="${MEMORY_BUDGET_PERCENT:-70}"

case "$native_auto" in
  0|1) ;;
  *) echo "erro: NATIVE_AUTO_MEMORY deve ser 0 ou 1" >&2; exit 2 ;;
esac

if [ "$native_auto" = 1 ]; then
  if [ -n "$requested" ]; then
    echo "erro: NATIVE_AUTO_MEMORY=1 não pode ser combinado com MAX_MBYTES_PER_BATCH" >&2
    exit 2
  fi
  echo "memória: modo histórico nativo-auto explicitamente habilitado" >&2
  echo 0
  exit 0
fi

if [ -n "$requested" ]; then
  if [[ ! "$requested" =~ ^[0-9]+$ ]] || [ "$requested" -le 0 ]; then
    echo "erro: MAX_MBYTES_PER_BATCH deve ser um inteiro positivo" >&2
    echo "      use NATIVE_AUTO_MEMORY=1 para reproduzir as políticas automáticas" >&2
    exit 2
  fi
  echo "memória: orçamento explícito de ${requested} MB" >&2
  echo "$requested"
  exit 0
fi

if [[ ! "$percent" =~ ^[0-9]+$ ]] || [ "$percent" -lt 1 ] || [ "$percent" -gt 95 ]; then
  echo "erro: MEMORY_BUDGET_PERCENT deve ser um inteiro entre 1 e 95" >&2
  exit 2
fi

free_mb="$(
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | sed -n '1p' | tr -d '[:space:]'
)"
if [[ ! "$free_mb" =~ ^[0-9]+$ ]] || [ "$free_mb" -le 0 ]; then
  echo "erro: não foi possível obter memory.free via nvidia-smi: '$free_mb'" >&2
  exit 2
fi

budget=$((free_mb * percent / 100))
if [ "$budget" -le 0 ]; then
  echo "erro: orçamento derivado inválido (${budget} MB)" >&2
  exit 2
fi

echo "memória: ${budget} MB (${percent}% de ${free_mb} MB livres)" >&2
echo "$budget"
