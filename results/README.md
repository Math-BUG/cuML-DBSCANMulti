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
