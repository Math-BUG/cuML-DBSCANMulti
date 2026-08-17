# Artefatos de resultados

O Git ignora dados, binários e logs brutos, mas permite manifests e resumos pequenos neste
diretório. Um manifesto liga uma alegação aos artefatos externos por SHA-256 e segue
`schemas/experiment-manifest.schema.json`.

- `manifests/`: um JSON por execução ou conjunto indivisível de execuções;
- `summaries/`: CSV/JSON derivados, com a lista dos manifests de origem;
- demais arquivos em `results/`: ignorados por padrão.

`manifests/example-preliminary.json` é apenas um exemplo estrutural. Hashes zero, valores
`null` e `experiment_id` iniciado por `example-` nunca representam evidência científica.
O `results/validation-matrix.json` bruto também fica ignorado até ser revisado; um manifesto
citável registra seu SHA-256, a identidade do binário e a cobertura semântica.

O executável calcula a mediana de `fit_ms_all`, mas as estatísticas de lote, rota e `nnz`
em `execution` pertencem somente à última repetição medida. O manifesto preserva essa
semântica em `protocol.execution_stats_scope="last_measured_repeat"`; não agregue esses
campos por inferência.

Para promoção, `validation_passed`, `semantic_oracle_run` e `backend_matrix_complete`
devem ser verdadeiros, e `semantic_valid_configurations` deve cobrir toda a grade. ARI e
igualdade de partição são diagnósticos e não substituem os checks de *core*, componentes,
ruído e borda. Consulte `docs/reprodutibilidade.md` antes de promover um resultado.

## Campanha oficial

Os arquivos acima e o schema `experiment-manifest.schema.json` descrevem resultados
históricos/preliminares. A campanha oficial usa os contratos
[benchmark-campaign.schema.json](../schemas/benchmark-campaign.schema.json),
[benchmark-sample.schema.json](../schemas/benchmark-sample.schema.json) e
[benchmark-run-manifest.schema.json](../schemas/benchmark-run-manifest.schema.json). Os
specs versionados são [PILOT](../scripts/campaigns/pilot.json) e
[CORE](../scripts/campaigns/core.json); o primeiro valida o protocolo com duas amostras por
método e não é evidência conclusiva, enquanto o segundo prevê dez amostras. Não execute a
CORE antes de um PILOT completo, sem divergência semântica, com contagens observadas iguais
às planejadas e revisão humana dos artefatos.

As dez amostras CORE sao cinco pares simetricos, nao dez unidades inferenciais. O PILOT
possui somente um par e serve para validar o fluxo, nunca para concluir desempenho.

Por padrão, a execução fica fora do Git em
`$DBM_BASE/results/pilot-<timestamp>/`. O fluxo produz somente JSON, JSONL, CSV e logs de
texto — nenhum relatório HTML ou notebook — com esta estrutura:

```text
manifest.json
environment.json
cases.json
plan.json
source-tree-hash.txt
inputs/{campaign-spec.json,requirements.lock.txt,validation-matrix.json}
inputs/pilot-manifest.json        # somente na CORE
schemas/{benchmark-campaign.schema.json,benchmark-sample.schema.json,benchmark-run-manifest.schema.json}
datasets/hashes.json
datasets/metadata/*.json
validation/{identity-*.json,selftest-*.log,matrix.json,...}
raw/records/*.json
raw/runtime/*.json
raw/<slurm-job-id>.jsonl
summaries/{summary.json,summary.csv}
logs/{benchmark.stdout.log,benchmark.stderr.log}
logs/{job_<slurm-job-id>.out,job_<slurm-job-id>.err}
build/
labels/                         # somente reproduções de falhas
```

Cada JSON em `raw/records/` é uma amostra medida e validada contra o schema; o JSONL é
reconstruído atomicamente desses registros. Blocos com partições divergentes não entram
como amostras válidas, e seus rótulos são preservados apenas em `labels/`. O manifesto
registra spec, árvore, binário, dataset, lock, matriz semântica e artefatos por SHA-256,
além das contagens planejadas/observadas e falhas. `status="partial"` ou qualquer falha
impede promoção.

Cada amostra/metodo tem timeout fixo de 900 s. Um timeout preserva o artefato de falha,
exclui o caso incompleto e nao e convertido silenciosamente em medicao valida.

`fit_ms` e a unica medida usada nos ganhos: pontos ja residentes no dispositivo ate os
rotulos no dispositivo. H2D, D2H, setup e `end_to_end_ms` sao diagnosticos. Cada bloco
`forward` e seu `reverse` compartilham `pair_index`; para cada razao positiva, o valor do
par e `sqrt(ratio_forward * ratio_reverse)`. Mediana, media, desvio-padrao, extremos,
quartis e IQR sao calculados sobre pares, sem excluir outliers. O IC95% reamostra pares em
um percentile bootstrap deterministico da mediana, com 10.000 iteracoes. Assim, o PILOT
tem `n=1` nao conclusivo e a CORE tem `n=5` por metodo/caso, sem pooling entre casos.

Os resumos incluem `ganho_multi_puro`, `speedup_vs_cuml`, `annotated_vs_dense`,
`auto_efficiency` e `efficiency_per_configuration`. A variante `best_forced` escolhe a
melhor rota depois de observar ambas; e diagnostico pos-selecao e nunca inferencia
principal. A leitura principal usa a rota `auto` predefinida.

Antes da submissão, `requirements.lock.txt` deve coincidir exatamente com o ambiente. O
hash de `scripts/source_tree_hash.py` também deve ser exportado como
`EXPECTED_SOURCE_TREE_HASH`; qualquer alteração no escopo desse hash exige outro build e
outro gate semântico. O job oficial [bench_pilot.slm](../scripts/bench_pilot.slm) faz essas
checagens e só começa a medir depois que identidade, selftests e matriz semântica passam.
Os comandos completos de `plan`, `prepare` e `sbatch` estão na seção
[Campanha oficial PILOT/CORE](../README.md#campanha-oficial-pilotcore).

A CORE planejada contem 35 casos, 1.550 registros raw, 5.790 fits medidos e 11.580 fits
incluindo warmups. `index64_multi_minpts_l4` e `index64_multi_both_2x4_auto` sao
diagnosticos `int64`; nao alteram a inferencia principal `int32`. Os sete casos N=200k ou
D=64 usam `tier=stress` e ficam fora da inferencia CORE primaria.

Spec, lock, schemas, matriz semantica e cada JSON de metadados sao copiados para a propria
campanha. `datasets/hashes.json` preserva os metadados completos e os descritores SHA-256;
os grandes arquivos de pontos nao sao duplicados. Na CORE, o manifesto PILOT validado
tambem e copiado. O finalizador anexa os hashes de stdout/stderr ao manifesto depois que
as redirecoes fecham. Assim, contratos, ambiente e metadados ficam autocontidos e
auditaveis. Registros duplicados sao erro; um caso incompleto e excluido inteiro dos
agregados. Uma futura CORE exige `--allow-core` e `--pilot-manifest` completo, aprovado e
com o mesmo `source_tree_sha256`.
