#!/bin/bash
# Configura temporários de compilação/Python fora do /tmp compartilhado do nó.
# Deve ser carregado com `source` depois de entrar na raiz do projeto.

set -euo pipefail

DBM_TMP_BASE="${DBM_TMP_BASE:-$HOME/dados/dbscanmulti/tmp}"
mkdir -p "$DBM_TMP_BASE"
DBM_JOB_TMP="$(mktemp -d "$DBM_TMP_BASE/job-${SLURM_JOB_ID:-local}.XXXXXX")"
export TMPDIR="$DBM_JOB_TMP"

cleanup_dbm_job_tmp() {
  rm -rf -- "$DBM_JOB_TMP"
}
trap cleanup_dbm_job_tmp EXIT

echo "temporarios: $TMPDIR"
