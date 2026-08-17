# cuML-DBSCANMulti

DBSCAN em GPU com exploração paralela de ε e *minPts*, derivado do DBSCAN do
[cuML](https://github.com/rapidsai/cuml). Uma única execução multiparamétrica
(Multi-EPS, Multi-minPts, Multi-Both) é comparada com a execução equivalente de
chamadas individuais do `cuml.cluster.DBSCAN`.

## Estado

As três variantes saem de uma única função, `fit_multi`, que recebe *k* valores de ε e
*l* de *minPts*: Multi-EPS é `l = 1`, Multi-minPts é `k = 1`, Multi-Both é ambos > 1.

| Variante | Situação |
|----------|----------|
| Multi-minPts | Implementada; há registro local de `--selftest` em A100 nos dois backends |
| Multi-EPS | Implementada; há registro local de `--selftest` em A100 nos dois backends |
| Multi-Both | Implementada; a grade 3×3 do `--selftest` é exatamente este caso |

No job histórico 4909, os selftests separados de cuVS e `codes` registraram as mesmas
contagens de clusters e ruído; **não compararam os vetores de rótulos entre si**. O HEAD
agora faz essa comparação direta por partição canônica e repete `codes` três vezes para
detectar não determinismo, mas essa versão ainda não foi executada no cluster. Portanto há
capacidade de validação no código atual, não evidência GPU de que o novo gate passou.

A comparação de tempo e rótulos contra `cuml.cluster.DBSCAN` está implementada em
`tools/bench_vs_cuml.py` e nos jobs Slurm. Os números adiante são **medições preliminares**:
os logs que lhes deram origem ainda não são artefatos versionados e o ambiente completo
ainda precisa ser congelado em `requirements.lock.txt`. Portanto eles descrevem aquelas
execuções, não uma garantia de desempenho geral. O contrato para promover uma execução a
resultado citável está em [docs/reprodutibilidade.md](docs/reprodutibilidade.md).

### Como o multi entra no cuML

No backend cuVS, o princípio é **preservar o pipeline do cuML**, em particular a busca de
vizinhança, que é a peça mais cara e mais ajustada. O backend alternativo `codes` é uma
implementação independente para validação. Por lote, o cuML faz:

```
cuVS epsilon_neighborhood::compute  ->  adjacência densa + graus
AdjGraph (exclusive_scan + adj_to_csr)  ->  CSR do lote
weak_cc_batched + MergeLabels  ->  rótulos
```

Chamar isso *k* vezes repetiria a etapa cara — a distância par-a-par, O(N²·D) — *k* vezes.
Pela monotonicidade do raio (vizinho sob um raio menor é vizinho sob todos os maiores), o
CSR do **maior** ε já contém todos os pares de que qualquer ε menor precisa. A implementação
mede a densidade depois dessa primeira busca e escolhe, **por lote**, entre duas rotas
semanticamente equivalentes:

1. **rota anotada, para grafo esparso:** cada entrada do CSR recebe o índice do menor ε que
   a contém, O(nnz·D), e os CSRs menores saem por compactação, O(nnz) por raio;
2. **rota densa:** a anotação faria acessos aleatórios sobre um CSR grande demais; o
   resultado já obtido no maior ε é reutilizado, o cuVS calcula os demais raios e cada CSR
   é construído diretamente — uma chamada por ε no total.

Logo, compartilhar a busca uma única vez é a rota principal no regime seletivo do DBSCAN,
mas não é uma invariável do executável. A decisão adaptativa está em
`anotar_compensa`/`run_multi_grid_cuvs`, e o `--selftest` força as duas rotas para exigir
partições idênticas.

Com `k = 1` e `l = 1`, **no backend cuVS**, a busca e as etapas herdadas seguem o caminho
algorítmico do cuML. Isso não torna os tempos automaticamente equivalentes: o Python cuML
usa `int32` e uma política própria de memória, enquanto este binário pode escolher outro
índice/orçamento/batching. Uma comparação isolada só controla essa diferença quando as
escolhas efetivas também coincidem; o JSON registra isso em `batch_budget_protocol`.

O ganho da rota anotada depende de o maior ε ser seletivo: a relação relevante é entre
N²·D e nnz·D. Se o maior ε levar *nnz* para perto de N², a anotação se aproxima do custo de
uma passagem completa e a política adaptativa pode escolher a rota densa.

### Dois backends

| `--backend` | Busca de vizinhança | Depende de |
|---|---|---|
| `cuvs` (padrão) | `cuvs::neighbors::epsilon_neighborhood::compute`, a mesma chamada do cuML | `libcuvs` |
| `codes` | kernel próprio, grava o índice do menor ε por par numa matriz N×N de bytes | RAFT/RMM e as dependências comuns exigidas pelo build atual |

O `cuvs` é o caminho principal, porque é o que significa reaproveitar o cuML. O `codes`
existe por dois motivos: compila sem `libcuvs`, e é uma segunda implementação independente
para validação cruzada — as duas têm de concordar nas mesmas asserções exatas do
`--selftest`.

O `--selftest` atual cobre oito grupos de propriedades; alguns são condicionais ao backend
e os últimos exercitam caminhos que não aparecem num teste de caminho feliz:

1. as **duas monotonicidades** numa grade 3×3 — com ε fixo o ruído não diminui quando
   *minPts* cresce; com *minPts* fixo não aumenta quando ε cresce. São propriedades
   acopladas: a coluna quebra se a matriz de códigos errar o menor ε que contém o par, a
   linha quebra se as máscaras errarem o maior *minPts* em que o ponto ainda é *core*;
2. **comparação direta cuVS↔`codes` e determinismo**: no build cuVS, compara as nove
   partições canônicas com `codes` e repete `codes` três vezes. Este grupo ainda não tem
   execução registrada no cluster;
3. **múltiplos lotes**: a mesma grade com o orçamento de memória apertado a ponto de
   forçar 4 lotes, exigindo rótulos idênticos aos do lote único. É o que exercita a fusão
   de rótulos entre lotes, a parte mais delicada herdada do cuML, que em dados pequenos
   nunca roda;
4. **variação de D e de k**: D ∈ {2, 8, 16, 32, 33} × k ∈ {1, 3, 5}. No backend `codes`
   isso percorre os quatro kernels de registrador, o genérico e as três especializações em
   *MAX_K*; no `cuvs`, exercita a anotação com e sem a linha de consulta em memória
   compartilhada. As dimensões extras são preenchidas com zero, que não altera a distância
   — a resposta esperada continua sendo exatamente três clusters, para qualquer D;
5. **fallback de dimensão muito alta**: no backend `codes`, compara D = 33 com D = 8193,
   que excede o tile em memória compartilhada e força o caminho de fallback;
6. **tipo de índice**: a mesma grade com `int64`, exigindo rótulos idênticos aos de
   `int32`. O tipo é detalhe de representação, e como N do selftest é pequeno esta é a
   única oportunidade de exercitar a instanciação `int64` com uma resposta conhecida;
7. **correção de lote**: fornece um `neigh_per_row` deliberadamente otimista e exige que a
   recuperação reduza o lote sem mudar as partições;
8. **rotas do cuVS**: força tanto anotação/compactação quanto reconstrução densa por ε e
   exige rótulos idênticos configuração a configuração. Em `codes`, rota é
   **não aplicável** (`route_tested=false`), não uma aprovação trivial.

O `scripts/check_gpu.slm` atual roda o selftest cuVS — que faz a comparação direta com
`codes` — e também o selftest do binário `codes` isolado. Esse fluxo ainda precisa ser
executado e preservado no cluster para se tornar evidência do HEAD.

## Ambiente

Um venv só, no **nó de login** — é download e instalação, não precisa de GPU nem de fila:

Na primeira criação, instale o baseline e deixe o próprio cluster resolver as transitivas:

```bash
INSTALL_CUML=1 bash scripts/setup_env.sh  # gera requirements.lock.txt
source .venv/bin/activate
```

Revise e versione o `requirements.lock.txt` produzido. Depois disso, uma reprodução deve
instalar e conferir exatamente esse ambiente, sem nova resolução:

```bash
RECREATE_VENV=1 USE_LOCK=1 bash scripts/setup_env.sh
source .venv/bin/activate
```

O script carrega os módulos sozinho (`GCCcore/12.2.0`, `CUDA/12.6.0`) e define
`PYTHONNOUSERSITE=1`, para o venv não enxergar o `~/.local/.../site-packages` — mesmo
padrão do `setup_venv_cluster.sh` do INF-494, que já roda nesse cluster.

Sem `USE_LOCK=1`, o script instala as ferramentas e, com `INSTALL_CUML=1`, o RAPIDS; ao
fim grava `requirements.lock.txt`. Com `USE_LOCK=1`, instala somente o lock existente e
falha se `pip freeze` divergir. Não combine uma reprodução citável com resolução aberta.

Usa o `python3` padrão do cluster (3.10.12), sem `module load Python`. Por isso os pacotes
RAPIDS diretos ficam restritos à série **26.02** em `requirements-cuml.txt`: a série 26.02
é a última que suporta Python 3.10. O lock ainda precisa ser gerado, revisado e versionado
no ClusterGPU/UFV; até lá, o ambiente não é reproduzível a partir de um clone. Não fabrique
retrospectivamente versões transitivas fora do cluster.

Os wheels trazem os cabeçalhos C++ em `site-packages/{libraft,librmm,rapids_logger}/include`, e é por
isso que **não é preciso conda nem compilar o cuML**. O `librmm/include/rapids` ainda traz
o CCCL (cuda/std, thrust, cub) na versão exata contra a qual RAFT e RMM foram compilados,
evitando depender do CCCL do módulo CUDA.

Se a `HOME` tiver cota apertada (o RAPIDS passa de 3 GB), mantenha o mesmo `VENV_DIR` na
criação e nas reproduções:

```bash
VENV_DIR=~/dados/venvs/dbscanmulti INSTALL_CUML=1 bash scripts/setup_env.sh
VENV_DIR=~/dados/venvs/dbscanmulti RECREATE_VENV=1 USE_LOCK=1 bash scripts/setup_env.sh
```

Os jobs também evitam o `/tmp` compartilhado do nó: criam um diretório exclusivo em
`~/dados/dbscanmulti/tmp` e o removem ao terminar. Use `DBM_TMP_BASE=/outra/particao` para
alterar esse local.

Para compilar sem instalar os pacotes Python RAPIDS, um wheel é um zip:
`bash scripts/fetch_rapids_headers.sh` extrai `include/` e `lib64/` de
RAFT/RMM/cuVS/rapids_logger, sem
pip nem venv.

## Build

Compila com `nvcc` direto — sem o CMake do cuML e sem `libcuml++`.

Se a árvore tiver `.git`, o build registra o commit e o estado *dirty*. Em cópias do
cluster sem `.git`, registra automaticamente um SHA-256 determinístico dos fontes com
`revision_kind="source-tree-sha256"`; não é necessário inicializar um repositório no nó.

```bash
make check-headers                  # mostra onde achou RAFT/RMM/cuVS
make CUDA_ARCH=sm_80 LINK_RAFT=1    # A100
make selftest                       # verificação embutida (precisa de GPU)

make CUDA_ARCH=sm_80 BACKEND=codes  # sem libcuvs
```

Com o venv ativo, os caminhos de RAFT/RMM são descobertos sozinhos. Em conda, passe
`RAPIDS_INCLUDE=$CONDA_PREFIX/include`.

`librmm.so` entra no link — RMM não é *header-only*. O shim local substitui o cabeçalho de
logger gerado pelo CMake do cuML; `rapids-logger` ainda é necessário porque
`raft/core/logger.hpp` inclui seus headers, e `librapids_logger.so` entra no link.
`LINK_RAFT=1` acrescenta `libraft.so`, que puxa nccl/cublas/cusolver/cusparse; é a
configuração que fecha o link no ClusterGPU/UFV. `BACKEND=cuvs` (o padrão) acrescenta
`libcuvs.so`, que vem como dependência do `cuml-cu12`. O `-rpath` já aponta para os
`lib64/` dos wheels, então o binário roda sem mexer em `LD_LIBRARY_PATH`.

`CUDA_ARCH=native` (padrão) exige uma GPU visível no momento da compilação; em nó de
*login* sem GPU, passe a arquitetura explicitamente — o job Slurm detecta sozinho.

## Datasets

Os geradores são uma adaptação fiel de `cluster_ufv/datasets_sinteticos.py` de
[Math-BUG/SSCAD-2026](https://github.com/Math-BUG/SSCAD-2026) — mesmas famílias, sementes,
normalização min-max para `[0,1]` e heurísticas de parâmetro. Sem isso os tempos não
seriam comparáveis com os números já publicados.

Os datasets vão para a **partição de dados**, não para a HOME — a suíte completa passa de
2 GB, e muito mais se entrarem N de 512k e 1M. Os scripts Slurm usam
`~/dados/dbscanmulti/knn-sample-rank-v2/{data,results}` por padrão, sobreponível por
`DBM_BASE`, `DATA_DIR` ou `RESULTS_DIR`. O sufixo de protocolo impede colisão com a campanha
histórica. O gerador, quando chamado à mão, precisa do `--out-dir` apontando para lá.

```bash
export DBM_BASE=$HOME/dados/dbscanmulti/knn-sample-rank-v2

python tools/gerar_datasets.py --listar
python tools/gerar_datasets.py --dataset moons_16d --n 100000 --out-dir "$DBM_BASE/data"

python tools/gerar_datasets.py --dry-run --preset escala   # quantos datasets e quanto disco
python tools/gerar_datasets.py --suite --preset escala --out-dir "$DBM_BASE/data"
```

`--preset` escolhe a grade da suíte: `artigo` (5 famílias × 8 dims × 8 N), `escala` (todas
as famílias × 2 dims × N de 4k a 1M, ~1,8 GB) ou `estresse` (só as famílias novas × 3
dims × N de 4k a 1M). `--familias`, `--dims` e `--ns` sobrepõem qualquer preset, e
`--dry-run` lista antes de gerar.

### Famílias acrescentadas para este trabalho

As 11 do catálogo original cobrem formatos (luas, anéis, espirais, blobs anisotrópicos).
Estas seis existem para estressar suposições específicas da execução multiparamétrica:

| Família | O que estressa |
|---|---|
| `nested_blobs` | Dois níveis de agrupamento: 3 super-grupos × 4 sub-blobs. **Nenhum ε único vê os dois** — raio pequeno dá 12 clusters, raio grande dá 3. É onde Multi-EPS entrega resultado diferente, não só tempo menor |
| `core_halo` | Núcleo denso + halo difuso no mesmo centro. A pertinência do halo depende do raio, então a grade de ε muda o **resultado** |
| `power_law_blobs` | Tamanhos em lei de potência: o grau por ponto varia ~10×, desbalanceando os kernels que usam um bloco por linha do CSR |
| `many_blobs` | Número de componentes cresce com N (até 512). Estressa `weak_cc` e a fusão de rótulos, que é o custo **por configuração** |
| `uniform` | Sem estrutura; *nnz* cresce suave com o raio, aproximando O(nnz·D) de O(N²·D). É o regime em que o ganho do multi-ε encolhe |
| `filaments` | Dimensão intrínseca 1 em D ambiente; o grau cresce ~linearmente com o raio. Regime de *nnz* oposto ao de `uniform` |

O `core_halo` já mostra o efeito na grade sugerida: os três ε saem em 0,027 / 0,098 / 0,120
— uma faixa de 4,5×, contra os ~20% típicos das outras famílias.

Duas ressalvas sobre os rótulos verdadeiros dessas famílias, para não confundir com defeito:

- **`uniform` não tem agrupamento verdadeiro.** Todos os pontos vêm rotulados como `-1`.
  ARI contra esse rótulo não significa nada; a família existe pelo perfil de *nnz*, não
  para medir qualidade;
- **`nested_blobs` não é para ser resolvido por um ε.** Com a grade padrão (quantis 0,50 /
  0,70 / 0,85) o melhor ARI contra os 12 sub-blobs fica em ~0,79, porque a faixa cai
  *entre* os dois níveis. Para ver os dois, alargue: `--eps-quantis 0.10,0.30,0.50,0.70,0.90`.
  Essa é exatamente a situação que justifica varrer ε em vez de escolher um.

`filaments` em 2D fica em ~0,82 porque seis retas com direções aleatórias no plano se
cruzam, e o cruzamento funde clusters. Em dimensão alta isso praticamente não ocorre — é
o mesmo fenômeno que o `chain_bridge` testa de propósito.

Cada dataset produz três arquivos:

| Arquivo | Conteúdo |
|---|---|
| `<nome>_n<N>.f32` | pontos, float32 *row-major* — entrada do binário |
| `<nome>_n<N>.labels.i32` | rótulos verdadeiros, int32 (`-1` = ruído) |
| `<nome>_n<N>.json` | N, D, semente, dimensão intrínseca e a grade `(eps, min_samples)` |

A grade sai do mesmo procedimento do artigo: *minPts* a partir da dimensão intrínseca
estimada por Levina-Bickel combinada com `log2(N)` (4 candidatos), e ε pelos quantis
0,50 / 0,70 e 0,85 da curva k-distância (3 candidatos) — a grade 3×4 = 12 configurações.
Já saem ordenados, que é a pré-condição do binário.

**Fronteira de protocolo:** a versão atual corrige o posto kNN pela fração amostral quando
`N > 60.000`; usar o mesmo posto numa amostra de 60 mil inflava ε para populações maiores.
O JSON novo registra população, tamanho da amostra, *minPts* populacional, posto amostral e
SHA-256 dos pontos, rótulos e gerador, além de
`"protocolo_dataset": "knn-sample-rank-v2"`. Todos os jobs validam esse identificador antes
de reutilizar um arquivo; metadados antigos ou uma grade de variantes diferente da esperada
encerram o job com erro. Datasets/grades anteriores a essa correção — inclusive
os usados nos jobs 4911–4917 — são protocolo histórico. A nova campanha deve regenerá-los,
obter hashes novos e não misturar as duas versões numa mesma agregação ou comparação direta
sem identificar explicitamente a mudança de protocolo.

O JSON alimenta tanto o nosso executável quanto o *baseline* cuML com **exatamente** os
mesmos parâmetros, que é o que torna a comparação de tempo honesta.

## Uso

Mantém compatibilidade com o contrato de linha de comando documentado pelo projeto
DBSCANMultiE, com `--min-samples` estendido para aceitar lista. Isso descreve uma interface;
não implica incorporação nem autorização para reutilizar o *harness* privado:

```bash
BIN="$(make -s print-target CUDA_ARCH=sm_80 LINK_RAFT=1)"
"$BIN" --input points.f32 --output labels.i32 \
       --n 100000 --d 16 --eps 0.25,0.35,0.5 --min-samples 5,10,20,40 --json

# faixa de eps, no mesmo formato do DBSCANMultiE
"$BIN" --input points.f32 --n 100000 --d 16 \
       --eps-min 0.1 --eps-max 0.5 --eps-step 0.1 --min-samples 5 --json
```

- `points.f32`: matriz *row-major* float32, sem cabeçalho;
- `labels.i32`: um int32 por ponto e por configuração, em ordem *config-major*;
- com *k* valores de ε e *l* de *minPts* saem `k*l` configurações, em ordem **eps-major**
  (`config = e*l + m`); os valores efetivos e sua ordem vão nos campos `eps`,
  `min_samples` e `config_order` do JSON;
- no máximo 16 valores de ε por execução;
- a última linha do stdout é o JSON com `fit_ms` e `configuration_count`;
- `fit_ms` cobre a execução do algoritmo e qualquer crescimento do workspace que ainda
  ocorra naquela chamada, e exclui leitura de arquivo e as transferências de entrada e
  saída; no protocolo oficial, o workspace externo cresce no *warmup* descartado e é
  reaproveitado na amostra medida;
- `--repeat R` mede R execuções e reporta a mediana em `fit_ms` (todas em `fit_ms_all`);
  `--warmup W` descarta as W primeiras. O bloco `execution` descreve somente a última
  repetição medida (`stats_scope="last_measured_repeat"`), não a agregação das R repetições;
  use `fit_ms_all` para auditar a mediana. A primeira chamada do processo paga o
  carregamento do módulo CUDA, que não é custo do algoritmo — sem `--warmup 1` a medição
  fica inflada.

## Campanha oficial PILOT/CORE

Os jobs `bench.slm`, `bench_variantes.slm`, `bench_escala.slm` e o uso direto de
`bench_vs_cuml.py` permanecem úteis para diagnóstico, mas são **protocolos preliminares**.
A campanha citável é definida pelos specs imutáveis
[PILOT](scripts/campaigns/pilot.json) e [CORE](scripts/campaigns/core.json), pelos schemas
em [schemas/](schemas/) e pelo *harness* [benchmark_campaign.py](tools/benchmark_campaign.py).
O PILOT tem sete casos e duas amostras medidas por método: verifica o fluxo completo em
regimes esparso, intermediário e denso, com casos escalar, Multi-minPts, Multi-EPS e
Multi-Both. Duas amostras servem para validar estrutura, semântica, memória e duração;
**não sustentam conclusão de desempenho**. A CORE amplia a grade e usa dez amostras por
método, mas não deve ser executada antes de o manifesto do PILOT estar completo, sem
falhas e revisado. O *harness* também bloqueia uma spec `phase=core` sem liberação explícita;
essa liberação não substitui a revisão do PILOT.

O protocolo oficial fixa A100-SXM4-80GB, backend cuVS, `int32`, `neigh_per_row=0`, orçamento
de 56.000 MB decimais (56.000.000.000 bytes), um *warmup* descartado e blocos simétricos: a ordem dos métodos é rotacionada
entre casos e cada bloco direto é seguido pela ordem inversa. Cada método roda em processo
isolado, evitando manter ao mesmo tempo os contextos CUDA do binário e do Python/cuML. As
rotas `annotated` e `dense` são diagnósticos forçados; `auto` é a política adaptativa que
representa o uso normal.

Cada amostra/metodo tem timeout fixo de 900 s. Timeout vira falha preservada do caso; os
demais casos continuam, e uma campanha parcial nunca libera a CORE.

Cada bloco `forward` e o bloco `reverse` seguinte compartilham o mesmo `pair_index` e
formam um par simetrico. A inversao da ordem reduz o vies temporal; os dois blocos nao sao
tratados como duas unidades inferenciais independentes.

A CORE planejada tem 35 casos, 1.550 registros medidos e dez amostras por metodo. Dois
casos `auto` isolam o custo de `int64`, sem multiplicar a matriz principal; sete casos
N=200k/D=64 usam `tier=stress` e nao entram na inferencia CORE primaria. A execucao CORE
exige simultaneamente `--allow-core` e `--pilot-manifest`; o manifesto precisa ser o PILOT
oficial completo e pertencer ao mesmo SHA-256 da arvore.

Esses 35 casos representam 5.790 fits DBSCAN medidos e outros 5.790 fits de warmup:
11.580 fits no total. O protocolo continua `int32`; somente
`index64_multi_minpts_l4` e `index64_multi_both_2x4_auto` usam o override `int64`, como
diagnostico separado de custo de indice. Eles nao alteram a inferencia principal `int32`.

### Fronteira de tempo e métricas

A medida oficial é `fit_ms`, com fronteira
`device-resident-input-to-device-labels`: começa com os pontos já na GPU e termina quando
os rótulos de dispositivo estão prontos. Ficam fora dela geração/leitura do dataset,
criação do processo, H2D e D2H; `setup_ms`, `h2d_ms`, `d2h_ms` e `end_to_end_ms` são apenas
diagnósticos. O *warmup* descarta o carregamento inicial; no binário experimental, também
absorve o crescimento inicial do workspace que será reaproveitado na amostra medida.
Os tempos experimental e cuML usam eventos CUDA, uma configuração de cada vez nos sweeps
escalares, e o tempo do sweep é a soma dessas configurações.

As razoes positivas sao calculadas dentro de cada bloco. Para cada metrica e rota, o valor
do par e a media geometrica dos dois sentidos:
`ratio_pair = sqrt(ratio_forward * ratio_reverse)`. A agregacao usa `pair_index`, nunca o
bloco isolado, como unidade inferencial.

- `ganho_multi_puro(route) = experimental_sequential.fit_ms / multi(route).fit_ms`;
- `speedup_vs_cuml(route) = cuml_sequential.fit_ms / multi(route).fit_ms`, comparação entre
  implementações/APIs, não um ganho algorítmico puro;
- `annotated_vs_dense = multi(dense).fit_ms / multi(annotated).fit_ms`; valor maior que 1
  favorece a rota anotada;
- `auto_efficiency = min(multi(annotated), multi(dense)) / multi(auto)`; valor próximo de 1
  indica que a decisão automática ficou próxima da melhor rota forçada;
- `efficiency_per_configuration = ganho_multi_puro / (k*l)`, métrica auxiliar do ganho por
  configuração, não substituta do ganho total.

`best_forced` escolhe a menor latencia entre `annotated` e `dense` depois de observar as
duas rotas. Por isso e um diagnostico pos-selecao potencialmente otimista; nunca e a
inferencia principal. A politica principal continua sendo `auto`.

O estimador oficial e a mediana dos valores por par. Media, desvio-padrao, minimo, maximo,
P25, P75 e IQR tambem sao calculados sobre esses valores, sem remover outliers. O IC95%
usa percentile bootstrap deterministico da mediana, com 10.000 reamostragens de pares.
O PILOT possui um unico par e e estruturalmente nao conclusivo. A CORE possui cinco pares
por metodo/caso; casos diferentes nunca sao agrupados para inflar `n`.

### Preparação e submissão do PILOT

No nó de login, depois de sincronizar a árvore final e ativar o ambiente travado:

```bash
cd ~/cuML-DBSCANMulti
source .venv/bin/activate
export DBM_BASE="$HOME/dados/dbscanmulti/official-campaign-v1"

python tools/benchmark_campaign.py plan \
  --spec scripts/campaigns/pilot.json
python tools/benchmark_campaign.py prepare \
  --spec scripts/campaigns/pilot.json \
  --data-dir "$DBM_BASE/data"

# Se setup_env.sh ainda nao o gerou, congele e revise este ambiente uma vez:
test -s requirements.lock.txt || python -m pip freeze --all > requirements.lock.txt
python -m pip freeze --all | diff -u requirements.lock.txt -
export LOCKFILE="$PWD/requirements.lock.txt"
export CAMPAIGN_DIR="$DBM_BASE/results/pilot-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$CAMPAIGN_DIR/logs"
export EXPECTED_SOURCE_TREE_HASH="$(python scripts/source_tree_hash.py)"

# Primeiro revalide o novo snapshot. Aguarde e confira job_<id>.out/.err.
sbatch scripts/check_gpu.slm

# Somente depois do novo check_gpu integralmente aprovado:
sbatch --export=ALL \
  --output="$CAMPAIGN_DIR/logs/job_%j.out" \
  --error="$CAMPAIGN_DIR/logs/job_%j.err" \
  scripts/bench_pilot.slm
```

`official-campaign-v1` e um diretorio novo, separado dos datasets historicos de tres
quantis. O comando `prepare` recusa qualquer arquivo cujo protocolo, grade, gerador ou
SHA-256 nao corresponda ao spec oficial; ele nunca reaproveita silenciosamente esses dados.

`prepare` publica datasets atomicamente e recusa metadados/hashes incompatíveis com
`knn-sample-rank-v2`; o job não gera datasets. `requirements.lock.txt` precisa existir,
estar revisado e coincidir exatamente com o ambiente, e seu SHA-256 entra nos artefatos.
Se o venv não for `.venv`, exporte `VENV_DIR` antes do `sbatch`.

O job 4996 validou semanticamente a revisão `02a77d03e738d23392f14a294e9ae208028e8e12`.
Como a instrumentação da campanha muda o hash da árvore, essa evidência não é reutilizada
silenciosamente: `bench_pilot.slm` exige `EXPECTED_SOURCE_TREE_HASH`, grava o mesmo hash no
build, confere a identidade dos binários, roda os dois selftests e uma nova matriz semântica
antes do benchmark. Qualquer nova mudança em `src/`, `tools/`, `scripts/`, `schemas/`, no
Makefile ou nos requisitos exige recalcular o hash e repetir esse gate. Planejar a CORE é
seguro e não usa GPU:

```bash
python tools/benchmark_campaign.py plan --spec scripts/campaigns/core.json
```

Mesmo depois de um PILOT aprovado, uma futura execucao CORE deve fornecer os dois flags
`--allow-core --pilot-manifest "$PILOT_CAMPAIGN_DIR/manifest.json"`. O harness verifica
`phase=pilot`, `status=completed`, ausencia de falhas, gate semantico, contagens completas,
hashes dos artefatos referenciados e igualdade do `source_tree_sha256` completo.

Não há submissão CORE autorizada nesta etapa. A organização dos artefatos e o critério de
promoção estão em [results/README.md](results/README.md).

## Comparação com o cuML

Os comandos desta seção pertencem ao protocolo histórico/preliminar. Para uma medição
oficial, use o PILOT acima e somente depois promova a campanha CORE.

```bash
sbatch scripts/bench.slm                                   # moons_16d, N=100k
sbatch --export=ALL,DATASET=blobs_32d,N=200000 scripts/bench.slm
sbatch --export=ALL,INDEX=int32 scripts/bench.slm          # força 5 lotes, para comparar
sbatch --export=ALL,NEIGH_PER_ROW=512 scripts/bench.slm    # tenta colapsar para 1 lote

sbatch scripts/bench_variantes.slm    # as três variantes separadas + varredura de k e l
sbatch scripts/bench_escala.slm       # ganho em função de N

# O Slurm interpreta vírgulas dentro de --export como separadores de variáveis.
export DATASET=core_halo_16d
export NS='4000,16000,64000,256000,1000000'
sbatch --export=ALL scripts/bench_escala.slm
unset DATASET NS
```

O `bench_variantes.slm` mede Multi-EPS, Multi-minPts e Multi-Both **com o mesmo número de
configurações**, que é a única comparação justa entre elas, e varre o número de
configurações para ajustar `S` e `L` por mínimos quadrados em vez de extrapolar de dois
pontos. As três compartilham coisas diferentes: em Multi-minPts o grau de um ponto não
depende de *minPts*, então a busca de vizinhança inteira é compartilhada e só a rotulagem
roda por configuração — é o teto. Multi-EPS paga anotação O(nnz·D) e compactação O(nnz)
por raio quando a rota esparsa vence; em lote denso, paga uma busca cuVS e a construção do
CSR por raio. O manifesto da execução deve registrar o perfil/rota usado para a leitura do
custo.

Ou direto, com o venv ativo e uma GPU disponível:

```bash
BIN="$(make -s print-target CUDA_ARCH=sm_80 LINK_RAFT=1)"
python tools/bench_vs_cuml.py --binario "$BIN" \
  --meta data/moons_16d_n100000.json --validar --imposto
```

As duas metades rodam no **mesmo job**, de propósito: o nó é compartilhado e o orçamento
de potência do chassi A100 SXM varia entre submissões, então comparar um tempo de hoje com
um de ontem mediria o vizinho, não o algoritmo. Ambos os lados cronometram só o ajuste com
os dados já na GPU, com `cudaEvent`, mesmo *warmup* e mediana de `--repeat` execuções. O
baseline recebe `calc_core_sample_indices=False`, porque também não calculamos esses
índices — cobrá-los dele inflaria o ganho.

`--imposto` roda **uma** configuração de cada lado e devolve `T = nosso / cuML`. `T` inclui
qualquer diferença de índice, orçamento e batching que permaneça entre o binário e a API
Python; não deve ser descrito apenas como “imposto do runner”. O campo
`batch_budget_protocol` registra se o pedido de memória foi controlado dos dois lados e
deixa explícito que o orçamento efetivo interno do cuML não é observável. Já
`ganho_multi_puro` compara a grade com o **mesmo binário** uma configuração por vez, sob as
mesmas escolhas, e é a alegação causal mais limpa sobre compartilhamento multiparamétrico.

### Evidência histórica/preliminar (logs locais 4909 e 4911–4917)

Esta é a série local mais recente disponível para inspeção, mas os logs não registram SHA
do commit nem hash/`build_id` do binário. Portanto **não provam o comportamento do HEAD**,
não satisfazem o manifesto e não devem ser promovidos a resultado reproduzível. Também
antecedem o gate por oráculo semântico descrito abaixo e a correção do posto kNN para
`N > 60.000`; suas grades não são diretamente comparáveis às que o gerador atual produzirá.

| Recorte histórico | Evidência registrada |
|---|---|
| Jobs 4911–4916 | 612 grades / 7.344 configurações; 98,434% de partições exatas; todas as máscaras e contagens de ruído/clusters iguais; ARI mínimo 0,999234; speedup mediano 4,6778×; ganho multi puro mediano 3,6111× |
| Job 4917, variantes | 47/48 partições exatas; ARI mínimo 0,999855; ganhos multi puros: Multi-EPS (8 ε) 6,099×, Multi-minPts (8 valores) 7,375× e Multi-Both 6,688× |

Essas observações permitem dizer apenas que aqueles executáveis produziram alta
concordância com o cuML naqueles casos e tempos. Máscaras/contagens iguais e ARI alto não
substituem a verificação independente de *core*, componentes, ruído e borda, e os speedups
não se generalizam além dos jobs registrados.

### Diagnóstico histórico: reaproveitar o workspace entre chamadas

Os números desta subseção vêm dos jobs 4862–4865 e permanecem somente como trilha do
diagnóstico que motivou o workspace externo; não são a evidência principal de desempenho.
Chamar `fit_multi` repetidamente no mesmo processo degradava: a segunda chamada em diante
custava **2,3×** a primeira (jobs 4862 e 4863). Ficou evidente ao variar só *minPts*, que
não altera a busca de vizinhança — quatro configurações com trabalho idêntico deram 112,
248, 256 e 261 ms, num patamar plano.

Diagnóstico, por eliminação:

| Hipótese | Teste | Resultado |
|---|---|---|
| Térmico / potência do chassi | o cuML roda as 12 configurações seguidas num processo Python único | estável em ~140 ms — descartado |
| Construir um `raft::handle_t` por chamada | compartilhar o handle (`--solo-handle-unico`) | 2,30× igual — descartado |
| Alocar e liberar o workspace por chamada | reaproveitar o buffer | é a causa |

Com `--neigh-per-row 512` e N=100k o workspace passa de **10 GB** — a adjacência densa é
`N × batch_size` bytes — e ele é alocado dentro da região cronometrada. O cuML no mesmo
cenário aloca ~2,1 GB, porque o `int32` limita o lote, e por isso não sofre o mesmo efeito.

A correção usa o que o cuML já tinha: `run_multi_grid_*` devolve o tamanho necessário
quando recebe `workspace == NULL`, justamente para o chamador alocar uma vez. `fit_multi`
agora aceita `workspace_externo`, cresce sob demanda e nunca encolhe.

Medido (jobs 4864 e 4865, `many_blobs_16d`, N=100k, 12 configurações — os dois batem):

| Protocolo | degradação da 1ª para as demais | vs. processo novo |
|---|---|---|
| A — um processo por configuração | 1,04× / 1,05× | — |
| B — mesmo processo, tudo recriado | 2,32× / 2,37× | +130% |
| **C — workspace reaproveitado** | **0,99× / 0,99×** | **−13%** |

O −13% de C é o custo da alocação que ele deixa de pagar: **~14 ms** por chamada, para um
workspace de 10 GB.

**Se você chama `fit_multi` em laço, passe um `rmm::device_uvector<char>`.** Sem isso, cada
chamada paga uma alocação de dezenas de GB.

No binário, `--solo` já faz isso (handle e workspace compartilhados). `--solo-isolado`
recria tudo e existe só para reproduzir o defeito. No benchmark, `--imposto` usa o modo
corrigido; `--imposto-por-processo` mede uma configuração por processo, que é o protocolo
mais simétrico com o cuML quando o que se quer é o `T` — porque aí a alocação entra dos
dois lados, como no `cuml.cluster.DBSCAN`.

`scripts/check_protocolo.slm` mede os três e detecta regressão.

### Tamanho do lote: `--index` e `--neigh-per-row`

O lote do cuML é limitado por `MAX_LABEL / N`, porque o CSR de um lote indexa `N·batch`
elementos. Com `int32` isso trava o lote a partir de N ≈ 46341 — em N=100k dá 5 lotes, e o
próprio cuML avisa no log que um tipo maior seria melhor. Cada lote a mais custa uma
recomputação da vizinhança na segunda passagem **e** uma rodada de fusão de rótulos **por
configuração**.

- `--index auto` (padrão) usa `int64` quando `N² ≥ INT_MAX`. É o mesmo template do cuML.
- `--neigh-per-row V` dimensiona o lote supondo V vizinhos por linha em vez do pior caso N.
  É o parâmetro `neigh_per_row` que existe em `dbscan.cuh` do cuML com um
  `///@todo: expose neigh_per_row to the user` — aqui ele está exposto. Como V é só um
  palpite, o runner mede o `nnz` antes de alocar o CSR; se o lote não couber, reduz o teto
  pelo grau observado e tenta novamente (até três tentativas, com pior caso N na última
  correção). O JSON registra `attempts` e `batch_corrections`; uma entrada que não caiba
  nem com o lote reduzido termina com erro, nunca como execução parcial aprovada.

**Leitura honesta do speedup:** `cuml.cluster.DBSCAN` usa `int32` e não expõe nenhum dos
dois. Usá-los é uma vantagem que o usuário do cuML não tem, e o `--imposto` a captura — se
`T` cair abaixo de 1, um fator `1/T` do speedup vem daí e não do multi. Por isso o
`ganho_multi_puro`, que compara o binário com ele mesmo.

`--validar` reporta ARI, concordância de ruído e igualdade da partição canônica. Essas
métricas são diagnósticos úteis, mas **não são o gate científico de correção**: ARI alto
pode esconder uma violação local, e duas implementações corretas podem escolher componentes
diferentes para um ponto de borda adjacente a mais de um cluster.

### Gate científico de correção

Depois do build cuVS, rode em uma alocação com GPU e o ambiente cuML ativo:

```bash
make CUDA_ARCH=sm_80 LINK_RAFT=1
python tools/run_validation_matrix.py \
  --binary "$(make -s print-target CUDA_ARCH=sm_80 LINK_RAFT=1)" \
  --random-seeds 10
```

Esse é o gate rápido de desenvolvimento. Para a campanha de defesa/publicação, use
`--random-seeds 100`, preserve `results/validation-matrix.json` e seu SHA-256 no manifesto.
O programa termina com código 2 se algum caso falhar e grava reproduções mínimas em
`validation_failures/`.

A matriz confronta cuVS, `codes` e cuML; força as rotas cuVS anotada e densa; cobre
`int32`/`int64`, lote único e múltiplos lotes; e verifica determinismo. As execuções com
rota forçada existem **somente para validação**. Medições do protocolo principal devem usar
`--route auto`, registrar a rota efetivamente observada por lote e nunca misturar tempos da
matriz de correção com o benchmark adaptativo.

Para cada caso pequeno, um oráculo CPU independente constrói o grafo ε exato e exige:

- pontos *core* exatamente quando a vizinhança fechada tem pelo menos *minPts* elementos;
- um cluster distinto para cada componente conexo do subgrafo de pontos *core*;
- ruído somente quando um ponto não-*core* não tem vizinho *core*;
- ponto de borda atribuído a um componente *core* adjacente, aceitando a ambiguidade
  legítima quando há mais de um.

Rótulos já salvos também podem ser auditados sem GPU. A ordem é `eps-major`, com um
`int32` por ponto e configuração:

```bash
python tools/validate_dbscan_matrix.py \
  --input data.f32 --n 100 --d 2 --eps 0.2,0.3 --min-samples 4,8 \
  --labels-cuvs cuvs.i32 --labels-codes codes.i32 --labels-cuml cuml.i32 \
  --exigir-tres-fontes --out validation.json

# reproduz exatamente um caso salvo por qualquer gate de validação
python tools/validate_dbscan_matrix.py --artifact validation_failures/failure.json
```

O oráculo é denso de propósito e recusa `N > --oraculo-max-n` (padrão: 5000), em vez de
degradar silenciosamente para uma aproximação. Benchmarks grandes podem reportar
ARI/partição, mas a alegação de correção deve referenciar, no mesmo commit/build, uma matriz
pequena/adversarial aprovada pelo oráculo.

## Cluster (ClusterGPU/UFV)

```bash
sbatch scripts/check_gpu.slm   # compila e roda o selftest em um job curto
squeue --me
cat job_<jobid>.out
cat job_<jobid>.err            # erros de compilação aparecem aqui
```

Se o venv não estiver em `.venv`, exporte antes: `VENV_DIR=~/dados/venvs/dbscanmulti sbatch ...`
não funciona — passe pelo ambiente com `sbatch --export=ALL,VENV_DIR=...`.

Não rode o teste de GPU no nó de *login*.

## Organização

| Caminho | Conteúdo |
|---------|----------|
| [third_party/cuml/](third_party/cuml/) | Subconjunto **verbatim** do DBSCAN do cuML na tag `v26.02.00`. Nada é editado aqui. Proveniência em [VENDORED.md](third_party/cuml/VENDORED.md) |
| [src/multi/](src/multi/) | Runner multiparamétrico, *core points* multi-*minPts*, o invólucro do cuVS e a anotação/filtro do CSR por ε |
| [src/compat/](src/compat/) | *Shim* de cabeçalho do cuML que só existe no build do CMake deles |
| [src/main.cu](src/main.cu) | Executável de linha de comando |
| [scripts/](scripts/) | Jobs Slurm: `check_gpu.slm` (sanidade), `bench_pilot.slm` (campanha oficial) e jobs históricos/preliminares |
| [tools/](tools/) | Geração de datasets, campanha de benchmark, matriz de correção e oráculo DBSCAN offline |
| [docs/fontes-primarias.md](docs/fontes-primarias.md) | Fontes, versões fixadas, pontos de modificação, licenças e pendências |
| [docs/reprodutibilidade.md](docs/reprodutibilidade.md) | Critério de aceite, metadados e limites das alegações experimentais |
| [docs/prontidao-publicacao.md](docs/prontidao-publicacao.md) | Checklist P0/P1/P2 e alegações atualmente sustentadas |
| [docs/licenciamento-e-proveniencia.md](docs/licenciamento-e-proveniencia.md) | Estado jurídico por fonte e bloqueios externos ainda abertos |
| [schemas/](schemas/) | Schemas versionados do manifesto experimental e do estado de proveniência |
| [results/](results/) | Manifests e resumos pequenos revisados; dados e logs brutos continuam fora do Git |
| [NOTICE](NOTICE) | Atribuição Apache-2.0 e lista de arquivos derivados |

## Licença, citação e contribuição

O subconjunto em `third_party/cuml/` é Apache-2.0 e os arquivos derivados identificados no
`NOTICE` preservam os avisos correspondentes. **O repositório como um todo ainda não tem
uma licença raiz escolhida pelos autores**, e a autorização referente à fonte privada F2
continua pendente. `NOTICE` e cabeçalhos SPDX de arquivos individuais não substituem essa
decisão. Veja [docs/licenciamento-e-proveniencia.md](docs/licenciamento-e-proveniencia.md)
antes de redistribuir ou publicar uma release.

Para citar o software, use [CITATION.cff](CITATION.cff), revisando autores e versão no
momento da release. Para mudanças, consulte [CONTRIBUTING.md](CONTRIBUTING.md); o gate local
de metadados é `python scripts/check_repo_metadata.py`.
