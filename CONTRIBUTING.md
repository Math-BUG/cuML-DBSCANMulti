# Contribuindo

Este é um protótipo de pesquisa com dependências CUDA e obrigações de proveniência. Uma
mudança aceita deve preservar tanto a correção quanto a rastreabilidade experimental.

## Antes de alterar

- Leia `docs/fontes-primarias.md`, `docs/licenciamento-e-proveniencia.md` e
  `docs/reprodutibilidade.md`.
- Não copie código de F2 ou de outra fonte sem licença/autorização verificável.
- Não edite `third_party/cuml/`: a árvore é verbatim e conferida por blob SHA.
- Mantenha mudanças derivadas do cuML fora dessa árvore, com SPDX, copyright original,
  aviso de modificação e entrada correspondente no `NOTICE`.
- Não altere números do README sem manifesto e artefatos que expliquem sua origem.

## Verificações CPU

```bash
python -m compileall -q tools scripts tests
python -m pytest -q
python scripts/check_repo_metadata.py
for file in scripts/*.sh scripts/*.slm; do bash -n "$file"; done
git diff --check
```

O CI executa esse conjunto sem GPU. Ele não prova correção CUDA nem desempenho. Mudanças no
algoritmo, build ou protocolo também exigem `scripts/check_gpu.slm` no cluster e manifesto
da execução. O gate `--publication` é deliberadamente mais estrito e continuará falhando
enquanto licença/proveniência externas estiverem abertas.

## Verificações GPU e ambiente

A primeira criação completa no ClusterGPU/UFV usa
`INSTALL_CUML=1 bash scripts/setup_env.sh`, gera `requirements.lock.txt` e exige revisão
antes de versioná-lo. Para reproduzir o ambiente congelado, use
`RECREATE_VENV=1 USE_LOCK=1 bash scripts/setup_env.sh`; `USE_LOCK=1` falha se o ambiente
instalado divergir do lock.

Antes de benchmarkar uma mudança de algoritmo, build ou protocolo, execute:

```bash
make CUDA_ARCH=sm_80 LINK_RAFT=1
python tools/run_validation_matrix.py \
  --binary "$(make -s print-target CUDA_ARCH=sm_80 LINK_RAFT=1)" \
  --random-seeds 10
```

Use 100 sementes na campanha de defesa/publicação. Esse gate usa o oráculo semântico de
*core*, componentes, ruído e borda; ARI ou igualdade de partição isolados não bastam. As
rotas forçadas da matriz são testes de correção, não configurações para o benchmark
principal, que continua adaptativo (`--route auto`).

## Resultados

Dados, binários e logs brutos permanecem fora do Git. Manifests e resumos pequenos,
revisados e sem material privado podem ser adicionados em `results/`. Use SHA-256 para
ligá-los aos artefatos externos. Nunca marque um resultado como `publication-ready` apenas
para satisfazer o schema.

## Commits e revisão

Prefira commits pequenos por tema, documente a motivação e liste os testes executados. Não
reescreva a proveniência vendorizada ou o histórico público para simular uma sequência de
desenvolvimento que não ocorreu.
