# Checklist de prontidão para defesa e publicação

Este checklist separa existência de código, evidência científica e autorização para
distribuição. Um item só está concluído quando seu critério de aceite pode ser verificado a
partir de um clone e dos artefatos referenciados; relato verbal ou log sem identidade do
binário não fecha o item.

## P0 — bloqueia a alegação ou a publicação

| Item | Escopo | Critério de aceite |
|---|---|---|
| Licença do projeto | release pública | Autores escolhem e adicionam `LICENSE` na raiz; `NOTICE`, SPDX e `provenance/project-status.json` são revisados sem atribuir licença a material privado por inferência |
| Proveniência de F2 | defesa e release | Autorização/licença verificável **ou** auditoria documentada de implementação independente sem cópia; estado deixa de ser `unresolved` com evidência arquivada |
| Autoria e citação | defesa e release | Todos os autores/titulares confirmam nomes, ordem, afiliações, versão e data em `CITATION.cff` |
| Ambiente congelado | defesa e artigo | `INSTALL_CUML=1 bash scripts/setup_env.sh` gera o lock no ClusterGPU/UFV; lock é revisado/versionado; `RECREATE_VENV=1 USE_LOCK=1 bash scripts/setup_env.sh` instala e confere exatamente |
| Identidade código–binário | qualquer resultado | Checkout limpo e tag; manifesto contém commit, SHA-256 do binário, `build_id`, `git_sha`, `git_dirty=false`, comando/flags e a correspondência entre SHA compilado e commit |
| Gate científico | alegação de correção | No mesmo commit/build, `run_validation_matrix.py` passa com 100 sementes; matriz cuVS/codes/cuML completa; `semantic_oracle_run=true`; *core*, componentes, ruído e borda aprovados; `semantic_valid_configurations == configuration_count`; JSON e hashes preservados |
| Campanha de desempenho | alegação de speedup | Datasets regenerados após a correção kNN para N > 60k; hashes de dados/gerador; mesma versão/job; rota `auto`; índice e `batch_budget_protocol` registrados; `execution.stats_scope=last_measured_repeat` preservado sem tratar rota/lote/nnz como agregados; speedup causal só com escolhas controladas, senão qualificado como comparação do sistema; duas execuções; estatística/dispersão; artefatos preservados |
| Gate de promoção | defesa/publicação | `python scripts/check_repo_metadata.py --publication` termina com código zero sem alterar artificialmente o estado de proveniência |

As rotas `annotated` e `dense` forçadas pertencem ao gate de correção. A campanha principal
mede `--route auto` e registra a rota observada por lote; tempos forçados não podem ser
apresentados como desempenho da política adaptativa.

## P1 — necessário para revisão confiável

| Item | Critério de aceite |
|---|---|
| CI CPU | `pytest`, compilação Python, schemas, hashes do vendorizado, licenças, links, shell/Slurm, matriz de configuração do Makefile e whitespace passam em checkout limpo |
| Proveniência de datasets e scripts internos | F3/F4 possuem titulares, autorização e versão identificados; nenhum arquivo privado é publicado por acidente |
| Escopo experimental | Tabelas separam GPU, driver, CUDA, dimensão, N, família, ε/*minPts*, índice e batching; conclusões não extrapolam o domínio medido |
| Falhas reproduzíveis | Toda divergência preserva dataset/rótulos/contexto por hash e pode ser reexecutada com `tools/validate_dbscan_matrix.py --artifact ...` |
| Revisão independente | Outra pessoa recria o ambiente pelo lock, verifica os manifests/hashes e reproduz ao menos uma execução sem instruções orais |

## P2 — melhora preservação e alcance

| Item | Critério de aceite |
|---|---|
| Arquivamento | Release/tag e artefatos recebem identificador persistente; o registro aponta exatamente para os manifests publicados |
| Matriz ampliada | A campanha é repetida em outra GPU/arquitetura ou a limitação a A100 fica explícita no resumo e nas conclusões |
| Automação de artefatos | Hashes e manifests são produzidos pelo job, revisados antes de versionar e verificados automaticamente sem armazenar datasets grandes no Git |
| Manutenção de upstream | Processo de atualização repete hashes dos 22 blobs, revisão das derivações, lock, gate semântico e campanha; nenhuma versão é atualizada isoladamente |

## O que pode ser alegado hoje

Com a qualificação adequada, o repositório permite afirmar que:

- há uma implementação multiparamétrica derivada do pipeline DBSCAN do cuML `v26.02.00`,
  com política cuVS adaptativa por lote e backend independente `codes`;
- o subconjunto vendorizado tem origem, commit e hashes Git registrados;
- existem testes CPU, selftest CUDA e ferramentas para matriz GPU/oráculo semântico e
  reprodução offline de falhas;
- os logs locais 4911–4917 registram a evidência histórica resumida no README, somente para
  aqueles executáveis, execuções e versão anterior do protocolo de geração.

Ainda **não** é sustentado afirmar que:

- o HEAD atual passou a campanha GPU de 100 sementes ou é correto para toda entrada;
- os speedups históricos são reproduzíveis, generalizáveis ou pertencem ao HEAD, pois os
  logs não registram commit/hash/`build_id` do binário, não têm manifesto completo e usam
  grades anteriores à correção do posto kNN para populações acima de 60 mil;
- ARI ou partição quase/exatamente iguais provam a semântica DBSCAN;
- o repositório inteiro está licenciado para redistribuição, enquanto a licença raiz e F2
  permanecerem sem decisão verificável;
- existe ambiente publicável reproduzível antes de o lock ser gerado e conferido no
  ClusterGPU/UFV.

O protocolo detalhado está em [reprodutibilidade.md](reprodutibilidade.md), e o estado
jurídico em [licenciamento-e-proveniencia.md](licenciamento-e-proveniencia.md).
