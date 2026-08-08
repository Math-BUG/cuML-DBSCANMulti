# Fontes Primárias — cuML-DBSCANMulti

**Projeto:** DBSCAN em GPU com exploração paralela de ε e *minPts*, construído como
**derivação do DBSCAN do cuML** (RAPIDS), e não como kernel escrito do zero.

**Objetivo do projeto:** avaliar uma única execução multiparamétrica (Multi-EPS,
Multi-minPts, Multi-Both) contra a execução equivalente de *k*, *l* ou *k×l* chamadas
individuais do `cuml.cluster.DBSCAN`, medindo o ganho de tempo em GPU.

**Objetivo deste documento:** registrar de onde vem cada peça do trabalho, em que versão
exata, sob qual licença, o que será reaproveitado, o que será modificado e o que fica
fora de escopo. Serve como base de rastreabilidade para o artigo e para a auditoria de
reprodutibilidade.

Última atualização: 2026-08-04.

---

## 1. Resumo das fontes

| ID | Fonte | Papel no trabalho | Licença | Acesso |
|----|-------|-------------------|---------|--------|
| **F1** | [rapidsai/cuml](https://github.com/rapidsai/cuml) | Base de código a ser derivada **e** *baseline* de comparação | Apache-2.0 | Público |
| **F2** | [Morphy999/DBSCANMultiE](https://github.com/Morphy999/DBSCANMultiE) | Contrato de execução, *harness* de benchmark/validação e referência de projeto multi-eps | Sem licença declarada | **Privado** |
| **F3** | Artigo SSCAD 2026 + [Math-BUG/INF-494](https://github.com/Math-BUG/INF-494) | Especificação algorítmica (monotonicidade, *bit packing*, protocolo experimental) | Interna | Próprio grupo |
| **F4** | `IC/Estendido/ids_generalization_pipeline` (local) | Convenções de execução no ClusterGPU/UFV (Slurm) | Interna | Local |
| **F5** | [l3lackcurtains/fast-cuda-gpu-dbscan](https://github.com/l3lackcurtains/fast-cuda-gpu-dbscan) | Comparador opcional (CUDA-DClust+, HiPC 2021) | Verificar | Público |

As duas fontes primárias no sentido estrito são **F1** e **F2**. F3–F5 são fontes de
apoio: especificação, infraestrutura e comparador de terceiros.

---

## 2. F1 — rapidsai/cuml

### 2.1 Identificação e *pin* de versão

| Campo | Valor |
|-------|-------|
| URL | https://github.com/rapidsai/cuml |
| Licença | Apache-2.0 (`SPDX-License-Identifier: Apache-2.0` em todos os arquivos do DBSCAN) |
| Copyright | NVIDIA CORPORATION, 2018–2026 |
| **Versão fixada** | **`v26.02.00`** — commit `22b12c8c3e378f17f35107f7fb4ffd65a3dce534`, 2026-02-05 |
| `main` na data desta consulta | `3d029900beb0`, 2026-08-04 (referência; **não** usar para os experimentos) |
| Cópia local | `third_party/cuml/` — 22 arquivos verbatim, conferidos por *blob SHA*; ver [`third_party/cuml/VENDORED.md`](../third_party/cuml/VENDORED.md) |

**Por que 26.02 e não a mais recente.** O `python3` do ClusterGPU/UFV é 3.10.12 e não há
módulo de Python mais novo. Os wheels do RAPIDS a partir de **26.04** exigem Python
≥ 3.11; o **26.02** ainda publica wheels `cp310`. Como o código derivado e o *baseline*
precisam ser da mesma versão, a tag do cuML acompanha essa escolha. O custo é baixo: no
subconjunto do DBSCAN, entre `v26.02.00` e `v26.06.00` mudaram apenas `runner.cuh` (ano
do copyright e um `#include <thrust/reduce.h>`) e `vertexdeg/algo.cuh` — exatamente os
dois arquivos que esta derivação substitui. Todo o resto é byte a byte idêntico.

**Regra:** todos os experimentos devem ser feitos contra **uma única tag fixa** do cuML,
usada simultaneamente como origem do código derivado e como *baseline* Python. Misturar
versões (código derivado de uma, baseline de outra) invalida a comparação. A tag
escolhida e o SHA devem ir para o CSV de resultados e para o artigo.

### 2.2 O que exatamente é reaproveitado

O DBSCAN do cuML é, em essência, um *pipeline* em cinco etapas executado em lotes de
linhas (`batch_size` calculado a partir da memória livre da GPU). Os arquivos relevantes:

| Arquivo em F1 | Papel | Reaproveitamento |
|---------------|-------|------------------|
| `cpp/include/cuml/cluster/dbscan.hpp` | API pública (`fit`, `EpsNnMethod`) | **Estender** com assinatura multiparamétrica |
| `cpp/src/dbscan/dbscan.cuh` | `dbscanFitImpl`, `compute_batch_size` | **Modificar** (orçamento de memória passa a depender de *k*) |
| `cpp/src/dbscan/runner.cuh` | Orquestra as 5 etapas, workspace, laço de lotes | **Modificar** (núcleo da mudança) |
| `cpp/src/dbscan/vertexdeg/algo.cuh` | Matriz de adjacência densa + graus de vértice | **Substituir** o cálculo de vizinhança |
| `cpp/src/dbscan/vertexdeg/pack.h`, `runner.cuh` | Empacotamento de parâmetros da etapa | **Modificar** (vetor de ε em vez de escalar) |
| `cpp/src/dbscan/corepoints/compute.cuh` | `mask[i] = vd[i] >= min_pts` | **Modificar** (l limiares sobre o mesmo `vd`) |
| `cpp/src/dbscan/adjgraph/algo.cuh` | `exclusive_scan(vd)` + `adj_to_csr` | **Reutilizar sem alterar**, chamado por configuração |
| `cpp/src/dbscan/mergelabels/*` | Fusão de rótulos entre lotes | **Reutilizar sem alterar**, por configuração |
| `python/cuml/cuml/cluster/dbscan.pyx` | Estimador Python | Referência da API; *baseline* de comparação |
| `cpp/tests/sg/dbscan_test.cu` | Testes do DBSCAN | Base para os testes de equivalência |
| `cpp/bench/sg/dbscan.cu` | Benchmark de referência | Referência de medição |

Fluxo original, por lote (verificado em `runner.cuh`):

1. `VertexDeg::run` → `cuvs::neighbors::epsilon_neighborhood::compute` produz `adj`
   (matriz booleana densa `batch_size × N`) e `vd` (graus). **É a passagem O(N·batch·D)
   que domina o tempo.**
2. `CorePoints::compute` → `core_pts[i] = vd[i] >= min_pts`.
3. `AdjGraph::run` → `adj` densa vira CSR (`ex_scan`, `adj_graph`).
4. `raft::sparse::weak_cc_batched` com filtro `core_pts` → rótulos do lote.
5. `MergeLabels::run` entre lotes; ao final, `final_relabel` + `relabelForSkl`
   (compatibilidade com scikit-learn: ruído = −1).

### 2.3 Pontos de modificação por variante

**Multi-minPts (mais barato).** `vd` **não depende de `min_pts`**. A etapa 1 roda uma
única vez; as etapas 2–5 rodam *l* vezes sobre o mesmo `vd`. Mudanças: `corepoints/compute.cuh`
escreve *l* máscaras (`core_pts[m][i]`), e o laço de `runner.cuh` itera as configurações.
Saída passa a ser `l × N` rótulos.

**Multi-EPS.** `cuvs::neighbors::epsilon_neighborhood::compute` só aceita ε escalar, o que
sugere trocá-la por um kernel próprio de múltiplos raios. **Essa foi a primeira decisão
tomada e ela estava errada**: é justamente a peça mais cara e mais ajustada do pipeline, e
substituí-la é o oposto de reaproveitar o cuML — além de tornar toda comparação de tempo
refém da qualidade do kernel novo.

A solução que preserva o cuML explora a monotonicidade do raio **depois** da busca, sobre o
CSR que o `AdjGraph` já constrói. Vizinho sob um raio menor é vizinho sob todos os maiores,
logo o CSR do **maior** ε contém todos os pares de que qualquer ε menor precisa:

1. `cuvs::…::compute` roda **uma vez**, no maior ε — chamada idêntica à do cuML;
2. `AdjGraph::run` do cuML produz o CSR do lote, sem alteração;
3. cada entrada do CSR é anotada com `e* = min{e : d ≤ ε_e}` — custo **O(nnz·D)**, porque
   só pares que já são vizinhos são visitados, não O(N²·D);
4. `vd[e][i]` sai do prefixo do histograma dos códigos da linha *i*, de graça no passo 3;
5. o CSR de cada ε menor sai por **compactação** das entradas com `código ≤ e` — O(nnz),
   sem nenhuma distância recalculada;
6. etapas 4–5 do fluxo original rodam por configuração, intocadas.

Com *k* = 1 os passos 3–5 desaparecem e o caminho é, kernel a kernel, o do cuML. O ganho é
a razão entre N²·D e nnz·D, e vale enquanto o maior ε for seletivo — se `nnz → N²`, a
anotação custa uma passagem completa, mas nesse regime todo ponto é vizinho de todos.

O kernel próprio de múltiplos raios (`src/multi/eps_neighborhood.cuh`, `--backend codes`)
foi mantido como backend alternativo: compila sem `libcuvs` e serve de segunda
implementação para validação cruzada, já que as duas têm de concordar nas asserções exatas
do `--selftest`.

**Multi-Both.** Combina os dois: contagem só em função de ε; comparação com os *l* valores
de *minPts* depois. São *k×l* rotulagens independentes, mas o CSR é compartilhado por
todos os *minPts* de um mesmo ε.

**O *bit packing* de F3 não se transfere — e isso é um achado, não uma omissão.** Na
Seção 3.2.3 do artigo ele existe porque lá o *union-find* decide a união **par a par**, e
por isso precisaria reler o vetor de pontos centrais a cada par; empacotar os limites de
*minPts* em 4 bits por ε num inteiro de 64 bits evita essa releitura. Nesta derivação a
conectividade não vem de uniões par a par: vem de um CSR construído uma vez por ε, e o
`weak_cc_batched` consulta a máscara de *core points* **uma vez por vértice**. O gargalo
que o empacotamento resolvia não existe aqui. Consequência prática: o limite de 16 ε e 15
*minPts* do artigo deixa de vir do empacotamento. O limite de 16 ε permanece, mas por
outra razão — o código do menor ε por par é um `uint8` e os contadores por raio ficam em
registradores; não há limite de *minPts* além da memória das máscaras.

**Impacto de memória (a validar antes de rodar N = 10⁶).** Em `dbscan.cuh:44` o custo por
linha do lote é `N·sizeof(bool) + (neigh_per_row+2)·sizeof(Index_)`. No backend `cuvs` o
termo denso é o mesmo do cuML (uma adjacência booleana, do maior raio); o que se acrescenta
é um segundo CSR (o filtrado) e um byte de código **por entrada**, não por par. No backend
`codes` a matriz de códigos é por par, `N` bytes por linha. Em ambos, as máscaras e os
rótulos passam a existir por configuração, então `compute_batch_size` recebe *k* e *l* —
senão o lote calculado estoura a memória da GPU.

### 2.4 Fora de escopo (versão 1)

Manter desabilitado e documentado, para não inflar a superfície de modificação:

- caminho **RBC** (`EpsNnMethod::RBC`, `cuvs::neighbors::ball_cover::eps_nn`) — só
  float/int64 e já cai para força bruta em vários casos;
- **OPG / multi-nó** (`opg = true`, `CorePoints::exchange`, `tree_reduction`);
- `sample_weight` (caminho `wght_sum`);
- métricas `Precomputed` e `CosineExpanded` — usar **L2 força bruta**, que é o caminho
  medido no artigo.

O *baseline* cuML deve ser executado com as mesmas restrições (mesma métrica, mesma
precisão float32, `algorithm="brute"`), senão a comparação mede coisas diferentes.

### 2.5 Estratégia de build

| Opção | Descrição | Custo | Entrega |
|-------|-----------|-------|---------|
| **A (recomendada)** | *Vendorizar* `cpp/src/dbscan/*` no repo, compilar com `nvcc` num binário próprio, usando apenas cabeçalhos de RAFT/RMM (o alvo `raft::raft` é *header-only*) | Baixo — minutos | Binário CLI, compatível com o contrato de F2 |
| **B** | Compilar `libcuml++` completo com o DBSCAN modificado + *binding* Python | Alto — build longo no cluster, resolução de dependências RAPIDS | `cuml.cluster.DBSCAN` com `fit_multi` |

**Viabilidade da opção A — confirmada pela auditoria de `#include` da árvore vendorizada.**
A única dependência compilada pesada é o **cuVS**, e ela aparece em **exatamente dois
arquivos**: `vertexdeg/algo.cuh` (`epsilon_neighborhood::compute`) e `runner.cuh` (índice
`ball_cover` do caminho RBC, fora de escopo). O restante do que se usa —
`raft::sparse::convert::adj_to_csr`, `raft::sparse::weak_cc_batched`,
`raft::label::make_monotonic`, `raft::linalg::*`, `rmm::device_uvector` — é *header-only*
(RAFT), mas **RMM e rapids_logger não são**: `librmm.so` e `librapids_logger.so` entram no
link.

O cuVS **permanece no build**, de propósito: é ele que faz a busca de vizinhança, e
reaproveitá-lo é o ponto do trabalho. Ele vem no wheel `libcuvs-cu12`, dependência do
`cuml-cu12` que o baseline já exige, então não custa nada de infraestrutura. O
`--backend codes` existe para o caso de o cuVS não estar disponível, e aí a árvore compila
só com RAFT/RMM.

Dois detalhes de compatibilidade já levantados:

- `cuml/common/logger.hpp` inclui `cuml/common/logger_macros.hpp`, que **não existe no
  repositório**: é gerado pelo CMake do cuML (`create_logger_macros(...)`,
  `cpp/CMakeLists.txt:259`). Fora do CMake do cuML é preciso um *shim* com
  `CUML_LOG_DEBUG` / `CUML_LOG_INFO` / `CUML_LOG_WARN` — podem ser no-ops.
- `cuml/common/utils.hpp` (macro `CUML_KERNEL`), `common/nvtx.hpp` e o `EpsNnMethod` de
  `cuml/cluster/dbscan.hpp` já estão vendorizados e são leves.

Confirmar no cluster: versões de RAFT/RMM disponíveis e compatíveis com CUDA 12.6.0
(módulo do ClusterGPU, ver F4).

**Onde ficam as modificações.** `third_party/cuml/` permanece **verbatim** — nada é
editado lá dentro. O código derivado e os *shims* ficam fora dessa pasta, de modo que
`diff -r` contra o upstream mostre exatamente a contribuição deste trabalho, que é
justamente o que o artigo precisa demonstrar.

### 2.6 Obrigações de licença (Apache-2.0)

Não é opcional e deve estar resolvido antes de publicar o repositório:

1. ✅ Cópia da licença Apache-2.0 incluída em `third_party/cuml/LICENSE`, junto do
   subconjunto vendorizado (§4a da licença).
2. ✅ `NOTICE` na raiz com a atribuição à NVIDIA, a versão fixada e a lista (por
   enquanto vazia) de arquivos derivados modificados.
3. ⬜ Preservar o cabeçalho SPDX e o *copyright* da NVIDIA em **todo** arquivo derivado,
   acrescentando uma linha de modificação — p. ex.
   `Modifications Copyright (c) 2026, Universidade Federal de Viçosa` (§4b) — e
   registrar o arquivo na lista do `NOTICE`. A fazer junto com a primeira modificação.
4. ⬜ Definir a licença do repositório. Recomendação: **Apache-2.0**, por compatibilidade
   direta com o código derivado. É decisão dos autores.
5. ⬜ No artigo, deixar explícito que a implementação **deriva** do cuML — isso é um ponto
   forte metodológico (mesmas estruturas, comparação mais justa), não uma ressalva.

---

## 3. F2 — Morphy999/DBSCANMultiE

### 3.1 Identificação e *pin* de versão

| Campo | Valor |
|-------|-------|
| URL | https://github.com/Morphy999/DBSCANMultiE |
| Visibilidade | **Privado** (acesso via `gh`; `WebFetch` retorna 404) |
| `main` na data desta consulta | `7b2631dfcf59`, 2026-07-29 |
| Licença | **Nenhuma declarada** |
| Linguagem | CUDA C++ (`dbscan.cu`, ~41 KB) + Python (harness) |

**Pendência bloqueante para publicação:** sem licença declarada, o código de F2 é, por
padrão, "todos os direitos reservados". Antes de reaproveitar qualquer trecho no
repositório público deste trabalho, obter do autor (a) autorização explícita e (b) uma
licença declarada no repositório de origem — de preferência Apache-2.0, para
compatibilidade com F1.

### 3.2 O que é reaproveitado

**(a) Contrato do executável CUDA** — é o que permite reusar o *harness* sem reescrevê-lo:

```text
./dbscan --input points.f32 --output labels.i32 --n N --d D \
         --eps E --min-samples M --json
```

- `points.f32`: matriz *row-major* sem cabeçalho, float32;
- `labels.i32`: um rótulo por ponto, `-1` para ruído;
- última linha do stdout: JSON com `{"fit_ms": ...}`;
- múltiplos ε: `--eps 0.1,0.2,0.5` ou faixa `--eps-min/--eps-max/--eps-step`
  (mutuamente exclusivos); saída com `E × N` rótulos em ordem **configuration-major**;
- `fit_ms` cobre a chamada completa do algoritmo (inclui *buffers* temporários e criação
  do *handle*), **exclui** leitura de arquivo e transferências; `e2e_ms` inclui processo e I/O.

**Extensão necessária para este trabalho:** `--min-samples` hoje é escalar. Passa a
aceitar lista, e a saída vira `k·l × N`. **Decisão registrada:** ordem *eps-major*,
`config_id = e·l + m`, preservando a convenção já existente quando `l = 1`. O JSON deve
reportar `configuration_count` (campo já lido por `compare_fast_dbscan.py`).

**(b) Harness de benchmark e validação** (`benchmark.py`, `plots.py`,
`profile_multi_eps.py`, `plot_multi_eps_profile.py`, `makefile`):

- soma dos *fits* escalares do cuML vs. um único *fit* multiparamétrico — exatamente a
  comparação de "trabalho equivalente" descrita no artigo;
- colunas `fit_speedup_vs_cuml` e `e2e_speedup_vs_cuml` (razão cuML/custom, `>1` = ganho),
  com as linhas do próprio cuML em `1.0`;
- validação estrita por **ARI** *e* **concordância de ruído** (ARI sozinho esconde
  divergências no rótulo `-1`); `ari_vs_cuml`/`noise_agreement` guardam o mínimo da
  configuração, `ari_by_eps`/`noise_agreement_by_eps` guardam os valores individuais;
- `min_ari: 0.999`, `min_noise_agreement: 0.999`, `warmup: 1`, `repeats: 5`, mediana nos
  gráficos; `make benchmark` falha em divergência, `make report` preserva no CSV;
- suítes já parametrizadas: `study=size` (D=16, N=1000…10000), `study=dimension` (N=5000,
  D=2…2048), `study=eps_count` (N=5000, D=16, E=1,3,5,9,17) e `benchmarks_scale_n.json`
  (D=2…256 × N=1000…10⁶, escalonável por `MAX_N`).

**(c) Referência de projeto** — o `dbscan.cu` de F2 resolve o mesmo problema por outro
caminho: duas passagens `cublasSgemm` em *tiles* para os produtos escalares (sem
materializar `N × N`), *union-find* em GPU (`find_root`, `union_components`,
`compress_multi_eps_paths`), histogramas acumulados por ε
(`accumulate_histograms_from_gram_tile`, `cumulative_histograms`) e memória temporária
≈ `12·E·N + 4·N + 4·T²` bytes. **Não será portado**: este trabalho parte do *pipeline* do
cuML. Serve como (i) referência de projeto para o kernel multi-ε, (ii) segunda
implementação para checagem cruzada de resultados, e (iii) baliza de desempenho —
se a versão derivada do cuML ficar bem abaixo dela, há regressão a investigar.

### 3.3 Divergência a resolver

O `makefile` de F2 assume `conda run -n dbscan_multi_env` e `LD_LIBRARY_PATH` de WSL
(`/usr/lib/wsl/lib`). O ClusterGPU/UFV (F4) usa módulos EasyBuild + `venv`. Parametrizar
o `makefile` (`RUN_PREFIX ?=`) em vez de fixar `conda`, e decidir como o cuML entra no
ambiente do cluster (wheel `cuml-cu12` no `.venv`, alinhado ao padrão de F4, ou Conda).

---

## 4. F3 — Artigo SSCAD 2026 e Math-BUG/INF-494

Fonte da **especificação algorítmica**. O que vem daqui:

- **Algoritmo 1** — DBSCAN paralelo em GPU com *union-find* em quatro etapas;
- **Algoritmo 2** — contagem de vizinhos para múltiplos ε: distância calculada uma vez,
  corte pelo maior ε, propagação do resultado para todos os raios compatíveis;
- **Algoritmo 3** — limite de *core point* por *minPts* explorando monotonicidade
  (uma vez que falha, não volta a ser central);
- **Bit packing** (Seção 3.2.3): 4 bits por valor de ε em inteiro de 64 bits → limite de
  **16 valores de ε e 15 de *minPts***;
- **Seleção de parâmetros**: *minPts* a partir da dimensão intrínseca (Levina-Bickel,
  MLE sobre os k vizinhos mais próximos) → 4 candidatos; ε pelos quantis 0,50 / 0,70 /
  0,85 da curva k-distância → 3 candidatos; grade 3×4 = 12 configurações;
- **Protocolo de medição**: 3 repetições, média e desvio, 1 execução de aquecimento
  descartada, `cudaEvent`, dados já residentes na GPU, float32;
- **Famílias de dados**: `dense_blobs`, `heterogeneous_blobs`, `moons`, `rings`, `spiral`,
  D ∈ {2,4,10,12,14,16,20,32}, N de 4 000 a 1 000 000.

**Resultados a superar/reproduzir** (A100-SXM4-80GB, CUDA 12.6): Multi-Both 28,6× em
N = 10⁶ contra 12 chamadas do cuML; média geral 10,9×; queda para 3,5× em D = 32; custo
da grade completa em 1,2–2,3× o custo de avaliar 3 ε isolados.

**Ponto em aberto herdado do artigo:** a equivalência de agrupamento (ARI/AMI) ficou fora
do escopo daquela versão. Aqui **entra no escopo desde o início**, via o protocolo de
validação de F2 (ARI + concordância de ruído contra o cuML, por configuração). Este é o
principal ganho metodológico de derivar do cuML em vez de reimplementar.

---

## 5. F4 — Infraestrutura de execução (ClusterGPU/UFV)

Fonte local: `C:\Users\AlphaXD\Desktop\UFV\IC\Estendido\ids_generalization_pipeline`.
Convenções extraídas de `scripts/run_debug_cluster_gpu.slm`, `scripts/run_ton_iot_gpu.slm`
e `scripts/check_gpu_rapids.slm`, a serem seguidas pelos scripts deste projeto:

```bash
#SBATCH --partition=scientific
#SBATCH --qos=scientific-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # 64 nos jobs maiores
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matheus.antony@ufv.br
```

- `set -euo pipefail` e *echo* de `SLURM_JOB_ID`, `SLURM_NODELIST`,
  `CUDA_VISIBLE_DEVICES`, `hostname`, `date`, `pwd`;
- módulos: `module --force purge` → `module load GCCcore/12.2.0` → `module load CUDA/12.6.0`;
  em seguida `nvidia-smi`;
- `PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"`, ativar `.venv` se existir;
- **dados, datasets e resultados em `$HOME/dados/...`**; a `HOME` fica só para scripts e
  configurações;
- submissão a partir do diretório do projeto: `sbatch scripts/<job>.slm`;
  acompanhamento com `squeue --me`; saídas em `job_<jobid>.out` / `.err`;
- **não** rodar teste de GPU no nó de *login* — usar um job curto de sanidade
  (o precedente é `check_gpu_rapids.slm`: 10 min, 4 CPUs, 16 GB, 1 GPU).

Consequência prática: a GPU do cluster **não é a A100-80GB do artigo**. O modelo, driver,
CUDA e capacidade de computação efetivamente usados devem ser registrados no CSV — os
números não são comparáveis diretamente aos de F3.

---

## 6. F5 — Comparador opcional: CUDA-DClust+

| Campo | Valor |
|-------|-------|
| URL | https://github.com/l3lackcurtains/fast-cuda-gpu-dbscan |
| Referência | Poudel, M.; Gowanlock, M. *CUDA-DClust+: Revisiting Early GPU-Accelerated DBSCAN Clustering Designs*. HiPC 2021 |
| Estado em F2 | Já integrado: `compare_fast_dbscan.py`, `fast_comparison_hipc2021.json`, alvos `fast-adapters` / `benchmark-fast` / `plot-three-way` |
| Limitação | Adaptadores compilados por dimensão fixa (`-DDIMENSION=2`, `-DDIMENSION=3`) — não cobre o estudo de dimensionalidade |

Uso previsto: comparação de três vias apenas em 2D/3D, como contexto. Não é o *baseline*
principal. Confirmar a licença do repositório antes de vendorizar qualquer adaptador.

---

## 7. Matriz de rastreabilidade

| Item do trabalho | Origem | Natureza |
|------------------|--------|----------|
| Pipeline DBSCAN em GPU (5 etapas, lotes, CSR, weak-CC, relabel sklearn) | F1 | Reuso direto |
| Baseline de comparação (`cuml.cluster.DBSCAN.fit`) | F1 | Reuso direto |
| Kernel de vizinhança multi-ε (código do menor ε por par) | F3 (Alg. 2) sobre *layout* de F1 | Contribuição |
| Máscaras de *core point* multi-*minPts* sobre `vd` único | F3 (Alg. 3) sobre F1 | Contribuição |
| Matriz de códigos (menor ε por par, 1 byte) + `codes_to_adj` por ε | Derivação de F3 (Alg. 2) para o *layout* de F1 | Contribuição |
| *Bit packing* dos limites de *minPts* por ε (64 bits, 4 bits/ε) | F3 (§3.2.3) | **Não aplicável** nesta derivação — ver §2.3 |
| Orçamento de memória e lote em função de *k* e *l* | F1 (`compute_batch_size`) | Modificação |
| Contrato CLI (`--input/--output/--eps/--min-samples/--json`) | F2 | Reuso + extensão (lista de *minPts*) |
| Ordem `config_id = e·l + m` (*eps-major*) | Decisão deste trabalho, estendendo F2 | Decisão |
| Harness de benchmark, CSV, gráficos, suítes de estudo | F2 | Reuso |
| Validação ARI + concordância de ruído por configuração | F2 | Reuso (escopo novo em relação a F3) |
| Seleção automática de ε e *minPts* (Levina-Bickel + quantis k-dist) | F3 | Reuso |
| Geradores sintéticos (5 famílias, D e N variáveis) | F3 | Reuso |
| Scripts Slurm, módulos, layout `~/dados` | F4 | Reuso de convenção |
| Comparação de três vias em 2D/3D | F5 via F2 | Opcional |

---

## 8. Metadados a registrar em cada execução

Sem estes campos, os resultados não são comparáveis entre execuções nem citáveis:

- modelo de GPU, versão do driver, versão do CUDA *toolkit*, *compute capability*;
- tag/SHA do cuML usada no código derivado **e** versão do pacote `cuml` do baseline;
- versões de RAFT/RMM/cuVS, se aplicável ao build escolhido;
- precisão (float32), métrica (L2), `algorithm`/`EpsNnMethod` do baseline;
- `n`, `d`, família do dataset, semente, lista de ε, lista de *minPts*, `k`, `l`;
- `warmup`, `repeats`, e a fronteira de medição (`fit_ms` vs `e2e_ms`);
- ARI e concordância de ruído por configuração;
- SHA do commit deste repositório.

---

## 9. Pendências e riscos

| # | Item | Tipo | Ação |
|---|------|------|------|
| 1 | F2 sem licença declarada | **Bloqueante para publicação** | Obter autorização escrita + licença no repositório de origem |
| 2 | Build A (vendor + `nvcc`) vs. B (`libcuml++` completo) | Decisão de arquitetura | **Resolvido: A.** Compila e linka no ClusterGPU/UFV contra os wheels `libraft`/`librmm`/`rapids_logger`/`libcuvs` |
| 2b | Substituir a busca de vizinhança do cuVS por kernel próprio | **Decisão revista** | **Revertida.** Era a peça mais ajustada do pipeline; o multi-ε passou a entrar sobre o CSR do cuML (§2.3). O kernel próprio virou `--backend codes`, para validação cruzada |
| 3 | Cabeçalhos de modificação + lista no `NOTICE` | Conformidade Apache-2.0 | **Feito** para `corepoints_multi.cuh` e `runner_multi.cuh`; repetir a cada novo arquivo derivado |
| 3b | *Shim* do logger (o `logger_macros.hpp` do cuML é gerado pelo CMake e não existe no repo) | Build | **Resolvido:** `src/compat/cuml/common/logger.hpp` substitui `<cuml/common/logger.hpp>` por prioridade de `-I` |
| 3d | Nada foi compilado ainda — não há CUDA toolkit na máquina de desenvolvimento | **Verificação** | Primeira compilação e `--selftest` via `sbatch scripts/check_gpu.slm` |
| 3c | Licença do próprio repositório | Decisão dos autores | Recomendado Apache-2.0 |
| 4 | `compute_batch_size` não considera *k* e *l* | Risco de OOM em N = 10⁶ | Modificar antes da suíte de escala |
| 5 | Conda (F2) vs. venv + módulos (F4) | Ambiente | Parametrizar `makefile`; decidir wheel `cuml-cu12` ou Conda |
| 6 | GPU do cluster ≠ A100-80GB do artigo | Comparabilidade | Registrar hardware; não comparar números absolutos com F3 |
| 7 | Adaptadores de F5 fixos em 2D/3D | Escopo | Restringir a comparação de três vias a 2D/3D |
| 8 | Limite de 16 ε / 15 *minPts* do *bit packing* | Limitação conhecida | Documentar; validar a grade 3×4 dentro do limite |

---

## 10. Como citar cada fonte

- **cuML / RAPIDS:** Raschka, S.; Patterson, J.; Nolet, C. *Machine learning in Python:
  main developments and technology trends in data science, machine learning, and
  artificial intelligence*. arXiv:2002.04803, 2020. Documentação:
  https://docs.rapids.ai/api/cuml/stable/api/generated/cuml.cluster.dbscan/
- **DBSCAN original:** Ester, M.; Kriegel, H.-P.; Sander, J.; Xu, X. *A density-based
  algorithm for discovering clusters in large spatial databases with noise*. KDD'96,
  p. 226–231.
- **Union-Find:** Tarjan, R. E. *Efficiency of a good but not linear set union algorithm*.
  JACM 22(2):215–225, 1975.
- **Dimensão intrínseca:** Levina, E.; Bickel, P. J. *Maximum likelihood estimation of
  intrinsic dimension*. NeurIPS 17, p. 777–784, 2005.
- **CUDA-DClust+:** Poudel, M.; Gowanlock, M. *CUDA-DClust+: Revisiting early
  GPU-accelerated DBSCAN clustering designs*. HiPC 2021.
- **G-DBSCAN:** Andrade, G. et al. *G-DBSCAN: a GPU accelerated algorithm for
  density-based clustering*. Procedia Computer Science, 18:369–378, 2013.
- **CUDA-DClust:** Böhm, C.; Noll, R.; Plant, C.; Wackersreuther, B. *Density-based
  clustering using graphics processors*. CIKM 2009, p. 661–670.
