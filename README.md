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
| Multi-minPts | Implementada; `--selftest` passa em GPU (A100) nos dois backends |
| Multi-EPS | Implementada; `--selftest` passa em GPU (A100) nos dois backends |
| Multi-Both | Implementada; a grade 3×3 do `--selftest` é exatamente este caso |

Os dois backends produzem **os mesmos rótulos**, configuração a configuração — duas buscas
de vizinhança independentes concordando nas mesmas asserções exatas.

Falta a comparação de tempo e de rótulos contra o `cuml.cluster.DBSCAN`, que é o que
`scripts/bench.slm` mede.

### Como o multi entra no cuML

O princípio é **não reescrever nenhuma peça do cuML**, em particular não a busca de
vizinhança, que é a mais cara e a mais ajustada. Por lote, o cuML faz:

```
cuVS epsilon_neighborhood::compute  ->  adjacência densa + graus
AdjGraph (exclusive_scan + adj_to_csr)  ->  CSR do lote
weak_cc_batched + MergeLabels  ->  rótulos
```

Chamar isso *k* vezes repetiria a etapa cara — a distância par-a-par, O(N²·D) — *k* vezes.
Pela monotonicidade do raio (vizinho sob um raio menor é vizinho sob todos os maiores), o
CSR do **maior** ε já contém todos os pares de que qualquer ε menor precisa. Então o
`cuvs` roda uma vez, no maior raio; cada entrada do CSR recebe o índice do menor ε que a
contém (custo O(nnz·D), porque só pares que já são vizinhos são visitados); e o CSR de cada
ε menor sai por **compactação**, O(nnz), sem recalcular distância nenhuma. Os graus em
todos os raios caem de graça, como prefixo do histograma dos códigos de cada linha, e
alimentam as máscaras de *core points* por *minPts*.

Com `k = 1` e `l = 1` o caminho é, kernel a kernel, o do cuML — o que torna a comparação
contra chamadas sequenciais uma medida honesta, sem imposto de implementação.

Isso vale enquanto o maior ε for seletivo, que é o regime de interesse do DBSCAN: o ganho é
a razão entre N²·D e nnz·D. Se o maior ε levar *nnz* para perto de N², a anotação custa o
mesmo que uma passagem completa — mas nesse regime todo ponto é vizinho de todos e o
agrupamento perdeu o sentido.

### Dois backends

| `--backend` | Busca de vizinhança | Depende de |
|---|---|---|
| `cuvs` (padrão) | `cuvs::neighbors::epsilon_neighborhood::compute`, a mesma chamada do cuML | `libcuvs` |
| `codes` | kernel próprio, grava o índice do menor ε por par numa matriz N×N de bytes | nada além de RAFT/RMM |

O `cuvs` é o caminho principal, porque é o que significa reaproveitar o cuML. O `codes`
existe por dois motivos: compila sem `libcuvs`, e é uma segunda implementação independente
para validação cruzada — as duas têm de concordar nas mesmas asserções exatas do
`--selftest`.

O `--selftest` cobre três coisas, e as duas últimas existem porque não apareceriam num
teste de caminho feliz:

1. as **duas monotonicidades** numa grade 3×3 — com ε fixo o ruído não diminui quando
   *minPts* cresce; com *minPts* fixo não aumenta quando ε cresce. São propriedades
   acopladas: a coluna quebra se a matriz de códigos errar o menor ε que contém o par, a
   linha quebra se as máscaras errarem o maior *minPts* em que o ponto ainda é *core*;
2. **múltiplos lotes**: a mesma grade com o orçamento de memória apertado a ponto de
   forçar 4 lotes, exigindo rótulos idênticos aos do lote único. É o que exercita a fusão
   de rótulos entre lotes, a parte mais delicada herdada do cuML, que em dados pequenos
   nunca roda;
3. **variação de D e de k**: D ∈ {2, 8, 16, 32, 33} × k ∈ {1, 3, 5}. No backend `codes`
   isso percorre os quatro kernels de registrador, o genérico e as três especializações em
   *MAX_K*; no `cuvs`, exercita a anotação com e sem a linha de consulta em memória
   compartilhada. As dimensões extras são preenchidas com zero, que não altera a distância
   — a resposta esperada continua sendo exatamente três clusters, para qualquer D;
4. **tipo de índice**: a mesma grade com `int64`, exigindo rótulos idênticos aos de
   `int32`. O tipo é detalhe de representação, e como N do selftest é pequeno esta é a
   única oportunidade de exercitar a instanciação `int64` com uma resposta conhecida.

O `scripts/check_gpu.slm` roda o `--selftest` nos **dois** backends, o que os torna
validação cruzada um do outro.

## Ambiente

Um venv só, no **nó de login** — é download e instalação, não precisa de GPU nem de fila:

```bash
bash scripts/setup_env.sh                 # ferramentas
INSTALL_CUML=1 bash scripts/setup_env.sh  # + baseline RAPIDS (download grande)
source .venv/bin/activate
```

O script carrega os módulos sozinho (`GCCcore/12.2.0`, `CUDA/12.6.0`) e define
`PYTHONNOUSERSITE=1`, para o venv não enxergar o `~/.local/.../site-packages` — mesmo
padrão do `setup_venv_cluster.sh` do INF-494, que já roda nesse cluster.

Um venv só, em duas etapas: as ferramentas primeiro, o RAPIDS depois. É a parte que mais
falha em cluster, e separando dá para gerar datasets, compilar e rodar o `--selftest`
mesmo se ela quebrar.

Usa o `python3` padrão do cluster (3.10.12), sem `module load Python`. Por isso o RAPIDS
fica na série **26.02**: a partir do 26.04 os wheels exigem Python 3.11.

Os wheels trazem os cabeçalhos C++ em `site-packages/{libraft,librmm}/include`, e é por
isso que **não é preciso conda nem compilar o cuML**. O `librmm/include/rapids` ainda traz
o CCCL (cuda/std, thrust, cub) na versão exata contra a qual RAFT e RMM foram compilados,
evitando depender do CCCL do módulo CUDA.

Se a `HOME` tiver cota apertada (o RAPIDS passa de 3 GB):

```bash
VENV_DIR=~/dados/venvs/dbscanmulti bash scripts/setup_env.sh
```

Para compilar sem instalar o RAPIDS, um wheel é um zip:
`bash scripts/fetch_rapids_headers.sh` extrai só o `include/`, sem pip.

## Build

Compila com `nvcc` direto — sem o CMake do cuML e sem `libcuml++`.

```bash
make check-headers                  # mostra onde achou RAFT/RMM/cuVS
make CUDA_ARCH=sm_80 LINK_RAFT=1    # A100
make selftest                       # verificação embutida (precisa de GPU)

make CUDA_ARCH=sm_80 BACKEND=codes  # sem libcuvs
```

Com o venv ativo, os caminhos de RAFT/RMM são descobertos sozinhos. Em conda, passe
`RAPIDS_INCLUDE=$CONDA_PREFIX/include`.

`librmm.so` e `librapids_logger.so` sempre entram no link — RMM não é header-only.
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
`~/dados/dbscanmulti/{data,results}` por padrão, sobreponível por `DBM_BASE`, `DATA_DIR` ou
`RESULTS_DIR`. O gerador, quando chamado à mão, precisa do `--out-dir` apontando para lá.

```bash
export DBM_BASE=$HOME/dados/dbscanmulti

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

O JSON alimenta tanto o nosso executável quanto o *baseline* cuML com **exatamente** os
mesmos parâmetros, que é o que torna a comparação de tempo honesta.

## Uso

Segue o contrato de linha de comando do DBSCANMultiE, para reaproveitar o *harness* de
benchmark e validação, com `--min-samples` estendido para aceitar lista:

```bash
./build/dbscan_multi --input points.f32 --output labels.i32 \
                     --n 100000 --d 16 --eps 0.25,0.35,0.5 --min-samples 5,10,20,40 --json

# faixa de eps, no mesmo formato do DBSCANMultiE
./build/dbscan_multi --input points.f32 --n 100000 --d 16 \
                     --eps-min 0.1 --eps-max 0.5 --eps-step 0.1 --min-samples 5 --json
```

- `points.f32`: matriz *row-major* float32, sem cabeçalho;
- `labels.i32`: um int32 por ponto e por configuração, em ordem *config-major*;
- com *k* valores de ε e *l* de *minPts* saem `k*l` configurações, em ordem **eps-major**
  (`config = e*l + m`); os valores efetivos e sua ordem vão nos campos `eps`,
  `min_samples` e `config_order` do JSON;
- no máximo 16 valores de ε por execução;
- a última linha do stdout é o JSON com `fit_ms` e `configuration_count`;
- `fit_ms` cobre a execução do algoritmo, incluindo a alocação do workspace, e exclui
  leitura de arquivo e as transferências de entrada e saída;
- `--repeat R` mede R execuções e reporta a mediana em `fit_ms` (todas em `fit_ms_all`);
  `--warmup W` descarta as W primeiras. A primeira chamada do processo paga o carregamento
  do módulo CUDA, que não é custo do algoritmo — sem `--warmup 1` a medição fica inflada.

## Comparação com o cuML

```bash
sbatch scripts/bench.slm                                   # moons_16d, N=100k
sbatch --export=ALL,DATASET=blobs_32d,N=200000 scripts/bench.slm
sbatch --export=ALL,INDEX=int32 scripts/bench.slm          # força 5 lotes, para comparar
sbatch --export=ALL,NEIGH_PER_ROW=512 scripts/bench.slm    # tenta colapsar para 1 lote

sbatch scripts/bench_variantes.slm    # as três variantes separadas + varredura de k e l
sbatch scripts/bench_escala.slm       # ganho em função de N
sbatch --export=ALL,DATASET=core_halo_16d,NS=4000,16000,64000,256000,1000000 scripts/bench_escala.slm
```

O `bench_variantes.slm` mede Multi-EPS, Multi-minPts e Multi-Both **com o mesmo número de
configurações**, que é a única comparação justa entre elas, e varre o número de
configurações para ajustar `S` e `L` por mínimos quadrados em vez de extrapolar de dois
pontos. As três compartilham coisas diferentes: em Multi-minPts o grau de um ponto não
depende de *minPts*, então a busca de vizinhança inteira é compartilhada e só a rotulagem
roda por configuração — é o teto; Multi-EPS ainda paga a anotação do CSR, O(nnz·D), e uma
compactação O(nnz) por raio.

Ou direto, com o venv ativo e uma GPU disponível:

```bash
python tools/bench_vs_cuml.py --meta data/moons_16d_n100000.json --validar --imposto
```

As duas metades rodam no **mesmo job**, de propósito: o nó é compartilhado e o orçamento
de potência do chassi A100 SXM varia entre submissões, então comparar um tempo de hoje com
um de ontem mediria o vizinho, não o algoritmo. Ambos os lados cronometram só o ajuste com
os dados já na GPU, com `cudaEvent`, mesmo *warmup* e mediana de `--repeat` execuções. O
baseline recebe `calc_core_sample_indices=False`, porque também não calculamos esses
índices — cobrá-los dele inflaria o ganho.

`--imposto` roda **uma** configuração de cada lado e devolve `T = nosso / cuML`. Como o
caminho k=1, l=1 é kernel a kernel o do cuML, T mede se sobrou trabalho extra no runner.
Ele também produz `ganho_multi_puro` — a grade de uma vez contra o **mesmo binário** rodando
uma configuração por vez. É o único número que isola o compartilhamento entre configurações
de qualquer outra diferença de ajuste.

### Resultado medido (A100, `moons_16d`, N=100k, grade 3×4)

| | `int32`, 5 lotes | `int64` + `--neigh-per-row 512`, 1 lote |
|---|---|---|
| Speedup contra o cuML | 7,46× | **11,03×** |
| Ganho do multi isolado | 7,77× | **8,46×** |
| Imposto (k=1, l=1) | 1,02× | **0,76×** |
| Partições idênticas | 12/12, ARI 1,000000 | 12/12, ARI 1,000000 |

As duas colunas mostram efeitos diferentes. O **ganho do multi isolado** compara o binário
com ele mesmo e é o que mede o compartilhamento entre configurações: 7,77× → 8,46×, porque
com um lote só desaparecem as fusões de rótulos entre lotes, que custavam por configuração.
O **imposto** de 0,76× diz que, com um lote, uma configuração nossa custa 112 ms contra 149
do cuML — 1,32× a mais de speedup que vem do tamanho do lote, não do multi. Os 11,03×
são o produto dos dois (8,46 × 1,32 ≈ 11,2, dentro do ruído do nó).

Ajustando `total = S + c·L`: com 5 lotes, `S ≈ 142 ms` e `L ≈ 8,3 ms` por configuração; com
1 lote, `S ≈ 107 ms` e `L ≈ 4,5 ms`. Ou seja, as fusões respondiam por cerca de metade do
custo por configuração. O teto do ganho do multi é `solo/L`, que sobe de ~18× para ~25×.

O imposto abaixo de 1 é um achado sobre o cuML, não um truque de medição: em N=100k ele
roda 5 lotes por causa do `int32` e da estimativa de pior caso, o que faz a busca de
vizinhança cobrir 1,93× mais pares do que os N² necessários. É exatamente o que o log dele
avisa e o que o `///@todo: expose neigh_per_row` antecipa.

### Reaproveitar o workspace entre chamadas

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
  `///@todo: expose neigh_per_row to the user` — aqui ele está exposto. Se V ficar muito
  abaixo do grau real, a alocação do CSR falha com `rmm::bad_alloc`, sem corromper nada.

**Leitura honesta do speedup:** `cuml.cluster.DBSCAN` usa `int32` e não expõe nenhum dos
dois. Usá-los é uma vantagem que o usuário do cuML não tem, e o `--imposto` a captura — se
`T` cair abaixo de 1, um fator `1/T` do speedup vem daí e não do multi. Por isso o
`ganho_multi_puro`, que compara o binário com ele mesmo.

`--validar` compara os rótulos com os do cuML configuração a configuração: ARI,
concordância de ruído e igualdade exata da partição (renumerando os clusters pela ordem de
primeira aparição, já que rótulos de DBSCAN são invariantes a permutação).

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
| [scripts/](scripts/) | Jobs Slurm: `check_gpu.slm` (sanidade), `bench.slm` (comparação) |
| [tools/](tools/) | Geração de datasets e o comparador contra o baseline cuML |
| [docs/fontes-primarias.md](docs/fontes-primarias.md) | Fontes, versões fixadas, pontos de modificação, licenças e pendências |
| [NOTICE](NOTICE) | Atribuição Apache-2.0 e lista de arquivos derivados |
