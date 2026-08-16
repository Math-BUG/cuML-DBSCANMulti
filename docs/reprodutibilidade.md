# Reprodutibilidade e promoção de resultados

Os JSONs emitidos por `tools/bench_vs_cuml.py` são resultados brutos. Um número só se torna
citável quando estiver ligado a código, ambiente, dados e validação por um manifesto que
satisfaça `schemas/experiment-manifest.schema.json`.

## Níveis de evidência

| Estado | Significado | Uso permitido |
|---|---|---|
| `preliminary` | execução local, com metadados ou artefatos incompletos | depuração e relato claramente qualificado |
| `reproducible` | artefatos, lock, dados e commit identificados; repetição independente concluída | relatório técnico limitado ao escopo registrado |
| `publication-ready` | requisitos anteriores, revisão de licença/proveniência e protocolo estatístico aprovados | tabela, artigo ou defesa |

O estado é propriedade do artefato, não do algoritmo em abstrato. Um resultado em uma A100
e um dataset não autoriza generalização para outras GPUs, famílias ou escalas.

## Procedimento

1. Trabalhe sobre checkout limpo e registre o SHA de 40 caracteres. Para publicação, use
   uma tag imutável.
2. Na primeira criação do ambiente no cluster, rode
   `INSTALL_CUML=1 bash scripts/setup_env.sh`. Revise e versione o
   `requirements.lock.txt` gerado; não reconstrua retrospectivamente versões ausentes.
   Reproduções posteriores devem usar
   `RECREATE_VENV=1 USE_LOCK=1 bash scripts/setup_env.sh`, que instala o lock e compara o
   `pip freeze`. O lock ainda não existe no repositório e continua sendo bloqueio externo.
3. Registre driver, toolkit CUDA, compute capability, compilador, flags, Python e versões
   cuML/CuPy/RAFT/RMM/cuVS.
4. Registre nome, `n`, `d`, semente e SHA-256 dos pontos, rótulos, metadados e gerador do
   dataset. Para `N > 60.000`, preserve também a população, tamanho da amostra e posto kNN
   corrigido usados na grade de ε.
5. Execute ambos os lados no mesmo job, com `warmup`, `repeat`, métrica, precisão, índice
   solicitado e efetivo, lote e `neigh_per_row` explícitos. Preserve o JSON bruto,
   stdout/stderr e SHA-256 do binário. Registre também `build_id`, `git_sha` e
   `git_dirty` emitidos pelo executável.
6. Registre se o perfil tomou rota anotada ou densa por lote. A implementação cuVS é
   adaptativa; não inferir a rota apenas pelo nome do backend. Para `codes`, registre
   `not-applicable` em vez de simular que um teste de rota passou. O bloco `execution` do
   JSON bruto tem `stats_scope="last_measured_repeat"`: lote, rota e `nnz` descrevem apenas
   a última repetição medida. Copie esse escopo para `protocol.execution_stats_scope` no
   manifesto e não o interprete como uma agregação das repetições. Preserve também
   `attempts`/`batch_corrections`: eles revelam quando um `neigh_per_row` otimista obrigou
   o runner a redimensionar o lote pelo grau observado.
7. Antes de medir, execute o gate semântico rápido descrito abaixo com 10 sementes. Na
   campanha de defesa/publicação, repita com 100 sementes e preserve o JSON da matriz.
8. Use ARI, concordância de ruído e partição canônica como diagnósticos secundários. O
   aceite científico vem do oráculo independente de estrutura DBSCAN, não dessas métricas.
9. Crie um manifesto em `results/manifests/`, compute hashes dos artefatos e declare as
   limitações da alegação.
10. Rode `python scripts/check_repo_metadata.py`. Para um candidato a publicação, rode
   também `python scripts/check_repo_metadata.py --publication`.

## Gate de correção antes do benchmark

Compile a configuração cuVS que contém os dois backends e consulte o alvo exato do build;
não use o alias mutável `build/dbscan_multi` em automações:

```bash
make CUDA_ARCH=sm_80 LINK_RAFT=1
python tools/run_validation_matrix.py \
  --binary "$(make -s print-target CUDA_ARCH=sm_80 LINK_RAFT=1)" \
  --random-seeds 10
```

Dez sementes são o gate rápido de desenvolvimento. Uma campanha candidata a
defesa/publicação usa `--random-seeds 100`, mantém o `results/validation-matrix.json`, os
artefatos de `validation_failures/` se houver falha e os respectivos hashes. O gate exige
determinismo, cuVS/codes/cuML, `int32`/`int64`, lote único/múltiplos lotes e as rotas cuVS
anotada/densa. Código de saída 2 significa reprovação.

As rotas forçadas fazem parte da **matriz de validação**, não do protocolo experimental
principal. O benchmark de desempenho deve usar a política adaptativa `--route auto` e
registrar `route_observed`; tempos obtidos ao forçar `annotated` ou `dense` não sustentam a
alegação de desempenho principal.

O manifesto preserva `batch_budget_protocol`. Se `max_mbytes_per_batch > 0`, o mesmo pedido
deve ser enviado aos dois lados e `controlled_equal_request=true`; mesmo assim, o orçamento
efetivo interno do Python cuML não é observável. Com orçamento automático, as políticas
diferem e o speedup contra cuML é uma comparação do sistema completo, não efeito causal
puro do multi. O mesmo vale se nosso índice efetivo não for `int32`, que é o tipo fixo da
API Python. `ganho_multi_puro`, por comparar o mesmo binário sob as mesmas escolhas, é a
medida causal mais limpa do compartilhamento entre configurações.

O oráculo CPU constrói, em float64, o grafo da vizinhança fechada para casos pequenos e
adversariais. Em cada combinação de ε e *minPts*, ele verifica diretamente:

1. pertencimento *core* pelo limiar de vizinhos;
2. componentes conexos do subgrafo *core*, sem fragmentar nem fundir componentes;
3. ruído matemático para pontos não-*core* sem vizinho *core*;
4. atribuição de borda somente a componente *core* adjacente, permitindo qualquer escolha
   entre componentes adjacentes quando a borda é genuinamente ambígua.

Essa quarta regra explica por que igualdade de partição não é uma definição suficiente de
correção. ARI próximo de 1 pode ocultar um erro; inversamente, partições diferentes podem
ser semanticamente válidas por uma escolha legítima de borda. O manifesto deve registrar
`semantic_oracle_run`, quantas configurações foram semanticamente válidas e se a matriz de
backends ficou completa.

Para auditar rótulos preservados sem GPU:

```bash
python tools/validate_dbscan_matrix.py \
  --input data.f32 --n 100 --d 2 --eps 0.2,0.3 --min-samples 4,8 \
  --labels-cuvs cuvs.i32 --labels-codes codes.i32 --labels-cuml cuml.i32 \
  --exigir-tres-fontes --out validation.json

python tools/validate_dbscan_matrix.py --artifact validation_failures/failure.json
```

O custo do grafo exato é quadrático. O limite padrão é `N = 5000`; excedê-lo produz erro,
não uma aprovação aproximada. Um benchmark de escala pode ser grande, desde que a correção
do mesmo commit e binário esteja ligada à campanha pequena/adversarial aprovada.

## Versão do protocolo de geração

O gerador atual corrige o posto kNN pela fração amostral quando a população excede a
amostra máxima de 60 mil pontos. Essa correção muda a grade de ε para `N > 60.000`. Assim,
datasets/grades produzidos antes dela, incluindo os associados aos logs 4911–4917, formam um
protocolo histórico separado. Antes da próxima campanha:

1. regenere os datasets com o gerador atual;
2. preserve os novos hashes de pontos, rótulos, JSON e gerador;
3. não agregue nem compare diretamente resultados antigos e novos sem estratificar a
   versão do protocolo;
4. execute novamente gate semântico e benchmark sobre os dados regenerados.

## Critérios mínimos de aceite

- `requirements.lock.txt` versionado e consistente com RAPIDS/cuML 26.02;
- repositório limpo e commit/tag presentes no manifesto;
- todos os campos de ambiente e dataset preenchidos, sem `null`;
- hashes de pontos, rótulos, metadados e gerador registrados; nenhuma mistura silenciosa
  entre grades anteriores e posteriores à correção do posto kNN;
- artefatos referenciados existentes e com SHA-256 conferido;
- SHA-256 do binário, `build_id`, `git_sha`, `git_dirty=false` e correspondência entre o
  commit compilado e o commit do manifesto;
- índice solicitado/efetivo e `batch_budget_protocol` completos; speedup apresentado como
  vantagem algorítmica somente quando as escolhas forem controladas, ou qualificado como
  comparação de políticas do sistema;
- `execution_stats_scope="last_measured_repeat"` preservado; estatísticas de lote, rota e
  `nnz` não apresentadas como se agregassem todas as repetições;
- pelo menos duas execuções independentes do protocolo declarado;
- estatística e dispersão reportadas, não apenas o melhor tempo;
- `validation_passed=true`, `semantic_oracle_run=true` e todos os checks de
  *core*/componentes/ruído/borda aprovados;
- `semantic_valid_configurations == configuration_count` e
  `backend_matrix_complete=true`;
- campanha de validação com pelo menos 100 sementes para `publication-ready`, sem reduzir
  o conjunto de casos depois de olhar o resultado;
- escopo da alegação restrito ao hardware, dados e parâmetros medidos;
- gate de licença/proveniência resolvido para `publication-ready`.

## Estado dos números atuais do README

Os logs locais registram execuções em A100 e sustentam seu uso como observações
preliminares. Eles são ignorados pelo Git, não possuem manifesto completo, SHA do commit,
hash/`build_id` do binário nem a nova campanha por oráculo semântico. Não provam o HEAD e
não permitem reprodução a partir de um clone. Por isso a tabela permanece explicitamente
histórica/preliminar até ser refeita ou promovida com todos os artefatos acima.

O checklist consolidado de bloqueios P0/P1/P2 e das alegações hoje sustentadas está em
[prontidao-publicacao.md](prontidao-publicacao.md).
