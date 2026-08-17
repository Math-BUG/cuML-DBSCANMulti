/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modifications Copyright (c) 2026, Universidade Federal de Viçosa (UFV).
 *
 * Derivado de third_party/cuml/cpp/src/dbscan/runner.cuh e de
 * third_party/cuml/cpp/src/dbscan/dbscan.cuh (cuML v26.02.00).
 *
 * Modificações em relação ao original:
 *
 *  1. Execução multiparamétrica: uma única chamada avalia a grade de k valores de ε por l
 *     valores de minPts. As três variantes saem da mesma função — Multi-EPS com l = 1,
 *     Multi-minPts com k = 1, Multi-Both com ambos > 1.
 *  2. Removidos os caminhos fora de escopo: RBC/ball_cover, multi-nó (opg), sample_weight,
 *     métricas Precomputed e Cosine. Só L2 força bruta.
 *  3. compute_batch_size passa a considerar k e l ao estimar a memória.
 *
 * A busca de vizinhança NÃO foi reescrita: continua sendo a chamada ao cuVS que o DBSCAN do
 * cuML faz (ver cuml_stages.cuh). A funcionalidade multi entra depois dela, sobre o CSR
 * que o AdjGraph do cuML constrói (ver csr_multi_eps.cuh). É onde está o ganho:
 *
 *   - a distância par-a-par, O(N² · D), roda uma vez por lote — no MAIOR raio, cujo CSR
 *     contém todos os pares de que os raios menores precisam;
 *   - o CSR de cada raio menor sai por compactação, O(nnz), sem recalcular distância;
 *   - os graus em todos os raios saem da anotação do CSR, O(nnz · D);
 *   - só a rotulagem (weak-CC + fusão) roda por configuração.
 *
 * Com k = 1 e l = 1 o caminho é, kernel a kernel, o do cuML — o que faz do custo desta
 * derivação uma medida honesta contra chamadas sequenciais.
 *
 * Existe ainda um segundo backend, o de src/multi/eps_neighborhood.cuh, com um kernel de
 * vizinhança próprio que grava o índice do menor ε por par. Ele não depende de libcuvs e
 * serve para validação cruzada, mas não é o caminho principal: reescrever a peça mais
 * ajustada do pipeline seria o oposto de reaproveitar o cuML.
 *
 * O que NÃO mudou, de propósito: a ordem das etapas, o weak_cc_batched com filtro de pontos
 * centrais, a fusão de rótulos entre lotes e o relabel final compatível com scikit-learn.
 * É o que mantém os rótulos comparáveis com os do cuML por ARI.
 */

#pragma once

#include "corepoints_multi.cuh"
#include "cuml_stages.cuh"
#include "eps_neighborhood.cuh"

#ifdef DBSCANMULTI_USE_CUVS
#include "csr_multi_eps.cuh"
#endif

#include <cuml/common/logger.hpp>

#include <raft/core/error.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cuda_dev_essentials.cuh>  // raft::alignTo
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace ML {
namespace Dbscan {
namespace Multi {

/**
 * Memória que cada linha do lote consome: adjacência booleana, código de ε, a fatia do CSR
 * e os k graus. Fatorada porque compute_batch_size_multi e max_bytes_for_batch_size têm de
 * usar exatamente a mesma conta — se divergirem, o teste de múltiplos lotes mede o lote
 * errado e passa sem exercitar nada.
 */
template <typename Index_ = int>
std::size_t est_mem_per_row_multi(Index_ N,
                                  int n_eps,
                                  bool use_cuvs      = false,
                                  Index_ neigh_hint  = 0)
{
  // Mesma convenção do cuML: `neigh_per_row <= 0` cai no pior caso, N vizinhos por linha.
  // O próprio cuML observa que esse pior caso quase nunca acontece — se ε é tão grande que
  // cada ponto se liga a 10% dos outros, o agrupamento já perdeu o sentido — e deixa um
  // TODO para expor o parâmetro. É o que `--neigh-per-row` faz.
  const std::size_t neigh_per_row =
    neigh_hint > 0 ? static_cast<std::size_t>(neigh_hint) : static_cast<std::size_t>(N);

  if (use_cuvs) {
    // Backend cuVS: adjacência densa do maior raio, o CSR do maior raio e — só quando há
    // mais de um raio — o CSR filtrado e um byte de código por entrada.
    const std::size_t por_entrada =
      (n_eps > 1 ? 2 : 1) * sizeof(Index_) + (n_eps > 1 ? sizeof(std::uint8_t) : 0);
    return static_cast<std::size_t>(N) * sizeof(bool) + neigh_per_row * por_entrada +
           2 * sizeof(Index_) + static_cast<std::size_t>(n_eps) * sizeof(Index_);
  }

  // Backend de códigos: adjacência booleana + um byte de código por PAR (não por entrada).
  return static_cast<std::size_t>(N) * sizeof(bool) +
         static_cast<std::size_t>(N) * sizeof(std::uint8_t) +
         (neigh_per_row + 2) * sizeof(Index_) + static_cast<std::size_t>(n_eps) * sizeof(Index_);
}

/**
 * Memória do workspace independente do lote: as k*l máscaras de pontos centrais e os
 * dois vetores temporários de rótulos usados pela fusão. O vetor de saída k*l*N é
 * alocado pelo chamador antes de `fit_multi` e, portanto, já foi descontado do valor
 * devolvido por `cudaMemGetInfo`; contá-lo aqui de novo reduziria artificialmente o lote.
 */
template <typename Index_ = int>
std::size_t est_mem_fixed_multi(Index_ N, int n_eps, int n_min_pts)
{
  const std::size_t n_configs = static_cast<std::size_t>(n_eps) * n_min_pts;
  return static_cast<std::size_t>(N) * sizeof(bool) * n_configs +
         static_cast<std::size_t>(N) * sizeof(Index_) * 2;
}

/**
 * O CSR de um lote com `nnz` entradas cabe na memória livre?
 *
 * `neigh_per_row` dimensiona o lote ANTES de a busca acontecer, então é uma promessa sobre
 * os dados, não um limite: se o grau real for maior, o lote já foi escolhido grande demais
 * e o CSR não cabe. Em heterogeneous_blobs o grau medido chegou a 56 mil vizinhos por
 * linha contra um palpite de 512, e o processo morria em `std::bad_alloc` sem dizer qual
 * parâmetro tinha mentido.
 *
 * O cuML tem exatamente a mesma estrutura (runner.cuh: `adj_graph.resize(maxadjlen)` com o
 * nnz medido depois da busca), mas nunca esbarra nisso porque o padrão dele é o pior caso,
 * N vizinhos por linha — o lote sai pequeno o bastante para qualquer densidade. Quem expõe
 * o parâmetro precisa checar.
 *
 * A margem de 10% cobre o pico do `resize`, que aloca o buffer novo antes de liberar o
 * antigo.
 */
/**
 * Teto artificial para o CSR, em bytes; 0 = usar a memória livre de verdade.
 *
 * Existe para o selftest: numa A100 de 80 GB nenhum dado pequeno o bastante para um teste
 * chega perto de estourar, então sem isso o caminho de correção do lote nunca roda e o
 * defeito que derrubou os jobs 4866-4871 voltaria sem ninguém notar.
 */
inline std::size_t& csr_teto_de_teste()
{
  static thread_local std::size_t teto = 0;
  return teto;
}

/**
 * Quantas vezes o lote precisou ser refeito. O selftest exige que seja > 0 quando aperta o
 * teto: sem isso, um dataset de teste pouco denso passaria pelo caminho normal e o teste
 * daria PASSOU sem ter exercitado nada.
 */
inline int& csr_correcoes_de_lote()
{
  static thread_local int n = 0;
  return n;
}

/** Rota efetivamente usada por um lote na última chamada. */
enum class BatchRoute : std::uint8_t { NotApplicable, Annotated, Dense };

inline const char* batch_route_name(BatchRoute route)
{
  switch (route) {
    case BatchRoute::Annotated: return "annotated";
    case BatchRoute::Dense: return "dense";
    default: return "not-applicable";
  }
}

/** Metadados da última chamada, usados para tornar cada JSON experimental auditável. */
struct ExecutionStats {
  std::size_t max_bytes_per_batch = 0;
  std::size_t batch_size          = 0;
  int batches                     = 0;
  int attempts                    = 0;
  int batch_corrections           = 0;
  int dense_batches               = 0;
  int annotated_batches           = 0;
  /** Uma entrada por lote, em ordem crescente do índice do lote. */
  std::vector<BatchRoute> batch_routes;
  long long max_nnz               = 0;
  long long total_nnz_max_eps     = 0;

  /** Zera a amostra sem liberar a capacidade adquirida pelo warmup. */
  void reset_preserving_capacity() noexcept
  {
    max_bytes_per_batch = 0;
    batch_size          = 0;
    batches             = 0;
    attempts            = 0;
    batch_corrections   = 0;
    dense_batches       = 0;
    annotated_batches   = 0;
    batch_routes.clear();
    max_nnz           = 0;
    total_nnz_max_eps = 0;
  }
};

inline ExecutionStats& last_execution_stats()
{
  static thread_local ExecutionStats stats;
  return stats;
}

/**
 * Bytes que cada entrada do CSR custa, por backend.
 *
 * No cuVS com mais de um raio são dois CSR (o do maior raio e o filtrado) mais um byte de
 * código por entrada. No backend de códigos há um CSR só — o código ali é por PAR e mora
 * no workspace, não por entrada. Usar a conta do cuVS para os dois superestimaria em 2,25x
 * e dispararia correções de lote desnecessárias.
 */
template <typename Index_>
std::size_t csr_bytes_por_entrada(int n_eps, bool use_cuvs)
{
  if (use_cuvs && n_eps > 1) return 2 * sizeof(Index_) + sizeof(std::uint8_t);
  return sizeof(Index_);
}

template <typename Index_>
bool csr_do_lote_cabe(Index_ nnz, std::size_t por_entrada)
{
  std::size_t disponivel = csr_teto_de_teste();
  if (disponivel == 0) {
    std::size_t total = 0;
    RAFT_CUDA_TRY(cudaMemGetInfo(&disponivel, &total));
  }

  const double preciso = 1.1 * static_cast<double>(nnz) * static_cast<double>(por_entrada);
  return preciso <= static_cast<double>(disponivel);
}

/** Grau médio observado num lote, arredondado para cima. */
template <typename Index_>
Index_ grau_medido(Index_ nnz, Index_ n_points)
{
  const std::size_t g =
    (static_cast<std::size_t>(nnz) + n_points - 1) / static_cast<std::size_t>(n_points);
  return static_cast<Index_>(std::max<std::size_t>(1, g));
}

/**
 * Quantas linhas de CSR cabem, dado o grau já medido. É o teto do lote depois de a busca
 * ter desmentido o palpite.
 *
 * Corrigir `neigh_per_row` não bastaria: ele entra em compute_batch_size_multi, que já
 * limita o lote a N — se o orçamento de memória não for o gargalo, o lote fica em N por
 * mais pessimista que o palpite fique, e a correção não sai do lugar. O teto age direto
 * sobre o lote, que é a única variável que realmente encolhe o CSR.
 *
 * A folga de 25% existe porque o grau medido é a média do lote que estourou, e o próximo
 * lote pode ser mais denso — em dados heterogêneos essa é a regra, não a exceção.
 */
template <typename Index_>
std::size_t linhas_de_csr_que_cabem(Index_ grau, std::size_t por_entrada)
{
  std::size_t disponivel = csr_teto_de_teste();
  if (disponivel == 0) {
    std::size_t total = 0;
    RAFT_CUDA_TRY(cudaMemGetInfo(&disponivel, &total));
  }

  const std::size_t por_linha = static_cast<std::size_t>(grau) * por_entrada;
  const std::size_t cabem     = static_cast<std::size_t>(0.8 * disponivel) / std::max<std::size_t>(1, por_linha);
  return std::max<std::size_t>(1, cabem);
}

/**
 * Tempo por fase, em ms. Só é preenchido quando `fit_multi` recebe um ponteiro.
 *
 * Existe porque a medição agregada não explicava um resultado: em heterogeneous_blobs_64d
 * uma configuração EXTRA na grade custava 1,46x uma execução independente inteira. Como a
 * grade faz estritamente menos busca de vizinhança que k*l execuções separadas, isso não
 * pode ser limite do método — é trabalho a mais em algum lugar, e sem separar as fases não
 * dá para saber onde.
 *
 * Cada fase é cercada por eventos CUDA com sincronização, o que serializa o stream: os
 * tempos SOMAM para perto do total, mas o total sob perfilamento é um pouco maior que sem.
 * Serve para comparar fases entre si, não para publicar número absoluto.
 */
struct PerfilMulti {
  double busca_p1 = 0, csr_p1 = 0, anotacao_p1 = 0, nucleo = 0;
  double busca_p2 = 0, csr_p2 = 0, anotacao_p2 = 0;
  double filtro = 0, rotulagem = 0, fusao = 0, relabel = 0;
  int n_lotes = 0;
  long long nnz_max = 0;

  /**
   * Rotulagem aberta por raio, com o nnz que cada chamada percorreu.
   *
   * O agregado não distingue duas explicações opostas para o custo por configuração: um
   * custo proporcional ao tamanho do CSR, ou um custo fixo por chamada ao weak_cc. Com o
   * nnz ao lado do tempo, ms por bilhão de arestas separa as duas — e é comparável entre a
   * grade e uma execução de uma configuração só, que é a comparação que interessa.
   *
   * `execucoes` conta as passagens de fit_multi acumuladas neste perfil (warmup inclusive),
   * para que os números possam ser normalizados por execução.
   */
  static constexpr int kMaxEpsPerfil = 16;
  double rotulagem_eps[kMaxEpsPerfil]   = {};
  long long nnz_rotulado[kMaxEpsPerfil] = {};
  int chamadas_eps[kMaxEpsPerfil]       = {};
  int execucoes                         = 0;

  /** Lotes que tomaram a rota densa (um CSR por raio) e lotes que anotaram. */
  int lotes_densos = 0, lotes_anotados = 0;

  double total() const
  {
    return busca_p1 + csr_p1 + anotacao_p1 + nucleo + busca_p2 + csr_p2 + anotacao_p2 +
           filtro + rotulagem + fusao + relabel;
  }
};

/** Acumula num campo do perfil o tempo de um escopo. Não faz nada se `alvo` for NULL. */
class Cronometro {
 public:
  Cronometro(double* alvo, cudaStream_t stream) : alvo_(alvo), stream_(stream)
  {
    if (alvo_ == nullptr) return;
    RAFT_CUDA_TRY(cudaEventCreate(&inicio_));
    RAFT_CUDA_TRY(cudaEventCreate(&fim_));
    RAFT_CUDA_TRY(cudaEventRecord(inicio_, stream_));
  }

  ~Cronometro()
  {
    if (alvo_ == nullptr) return;
    cudaEventRecord(fim_, stream_);
    cudaEventSynchronize(fim_);
    float ms = 0;
    cudaEventElapsedTime(&ms, inicio_, fim_);
    *alvo_ += ms;
    cudaEventDestroy(inicio_);
    cudaEventDestroy(fim_);
  }

  Cronometro(const Cronometro&)            = delete;
  Cronometro& operator=(const Cronometro&) = delete;

 private:
  double* alvo_;
  cudaStream_t stream_;
  cudaEvent_t inicio_{}, fim_{};
};

/** Endereço do campo, ou NULL quando não há perfil — para o Cronometro virar no-op. */
#define DBM_FASE(perfil, campo) ((perfil) ? &(perfil)->campo : nullptr)

/**
 * Quanto a busca do cuVS é mais eficiente, por operação, que a anotação do CSR.
 *
 * Medido no job 4895, heterogeneous_blobs_64d, N=64k, D=64:
 *
 *   busca (cuVS)   64000^2 x 64 = 2,6e11 operações em  120 ms  ->  2,2 T/s
 *   anotação        1,17e9 x 64 = 7,5e10 operações em 1374 ms  ->  0,055 T/s
 *
 * Fator 40. A causa é acesso, não cálculo: o cuVS varre os pares em tiles, com a fatia de x
 * em memória compartilhada; a anotação percorre o CSR e busca x[col] entrada por entrada —
 * 1,17e9 gathers aleatórios de 64 floats, 300 GB de tráfego a 218 GB/s, 11% do pico da
 * A100. Recalcular a distância varrendo em ordem sai mais barato que buscá-la esparsamente.
 */
constexpr long long kVantagemDoTiling = 40;

/**
 * Custo fixo de uma chamada ao cuVS mais o AdjGraph, em operações equivalentes de busca.
 *
 * Medido em dense_blobs_2d N=4000, D=2 (job 4900): a rota densa acrescentou 5 chamadas e
 * 65,2 ms — 13,0 ms cada — num caso em que a anotação inteira custava menos de 1 ms. Ali o
 * kernel da busca é 8 192x menor que em N=64000: o que sobra é lançamento, alocação
 * temporária e sincronização, tudo independente do tamanho.
 *
 * 13,0 ms na taxa medida da busca tiled (~4,6e12 operações/s) dão ~6e10 operações.
 */
constexpr long long kCustoFixoPorBusca = 60000000000LL;

/**
 * Compensa anotar o CSR, ou sai mais barato refazer a busca uma vez por raio?
 *
 * Anotar custa ~nnz*D*40 em operações equivalentes de busca (ver kVantagemDoTiling). Trocar
 * uma busca por k custa (k-1) vezes uma busca inteira, e cada busca traz um custo fixo além
 * do kernel.
 *
 * O termo fixo não é detalhe: sem ele a regra compara só trabalho assintótico e, em N
 * pequeno, troca ~0 ms de anotação por dezenas de ms de overhead. Foi o que fez
 * dense_blobs_2d N=4000 passar de 5,4 ms para 70,6 ms no job 4900 — a única regressão da
 * varredura inteira.
 *
 * Classificação que isto produz, conferida contra o medido:
 *
 *   dense_blobs_2d      N=4000  D=2   nnz 4e6     -> anota  (a rota densa custava 13x)
 *   filaments_64d       N=64k   D=64  nnz 1,34e7  -> anota  (ganha 6,4x anotando)
 *   heterogeneous_64d   N=16k   D=64  nnz 7,2e7   -> densa  (0,99x -> 1,56x)
 *   heterogeneous_64d   N=64k   D=64  nnz 1,17e9  -> densa  (0,78x -> 1,69x)
 *
 * @param nnz    entradas do CSR no maior raio, neste lote
 * @param pares  N * n_points, o total de pares que a busca varre neste lote
 * @param D      dimensão: entra dos dois lados, mas não se cancela contra o custo fixo
 */
inline bool anotar_compensa(long long nnz, long long pares, int n_eps, long long D)
{
  // A decisão só precisa da ordem entre os custos. Promover antes de multiplicar evita
  // overflow assinado (comportamento indefinido) em N/D grandes sem saturar a heurística.
  const long double custo_anotar =
    static_cast<long double>(nnz) * D * kVantagemDoTiling;
  const long double custo_buscas = static_cast<long double>(n_eps - 1) *
                                   (static_cast<long double>(pares) * D +
                                    static_cast<long double>(kCustoFixoPorBusca));
  return custo_anotar < custo_buscas;
}

/**
 * Força uma rota: 0 = decidir pelo nnz medido (padrão), 1 = sempre anotar, 2 = sempre densa.
 *
 * Existe para o selftest. As duas rotas calculam a mesma coisa por caminhos diferentes —
 * uma compacta o CSR do maior raio, a outra reconstrói um CSR por raio — e a única prova de
 * que a escolha é só desempenho, e não semântica, é rodar as duas nos mesmos dados e exigir
 * rótulos idênticos. Sem isto o teste veria só a rota que o dataset de teste calhar de
 * escolher.
 */
inline int& rota_forcada()
{
  static thread_local int rota = 0;
  return rota;
}

/**
 * Orçamento de memória que produz o tamanho de lote pedido — a inversa de
 * compute_batch_size_multi. Existe para o selftest conseguir forçar mais de um lote e
 * exercitar a fusão de rótulos, que de outra forma nunca roda em dados pequenos.
 */
template <typename Index_ = int>
std::size_t max_bytes_for_batch_size(Index_ N,
                                     int n_eps,
                                     int n_min_pts,
                                     std::size_t batch_size,
                                     bool use_cuvs     = false,
                                     Index_ neigh_hint = 0)
{
  return est_mem_fixed_multi<Index_>(N, n_eps, n_min_pts) +
         batch_size * est_mem_per_row_multi<Index_>(N, n_eps, use_cuvs, neigh_hint) + 1;
}

/**
 * Tamanho de lote, adaptado de ML::Dbscan::compute_batch_size (dbscan.cuh do cuML).
 *
 * Diferenças: a memória por linha inclui a matriz de códigos de ε (1 byte por par, além
 * da booleana de adjacência) e os k graus por ponto; a memória fixa inclui as k*l máscaras
 * de pontos centrais e os dois vetores temporários de rótulos. Os k*l rótulos de saída
 * já estão residentes fora do workspace. Sem essa separação, o modo automático desconta
 * a saída duas vezes da memória livre e escolhe lotes desnecessariamente pequenos.
 */
template <typename Index_ = int>
std::size_t compute_batch_size_multi(std::size_t& estimated_memory,
                                     Index_ N,
                                     int n_eps,
                                     int n_min_pts,
                                     std::size_t max_bytes_per_batch,
                                     bool use_cuvs     = false,
                                     Index_ neigh_hint = 0)
{
  const std::size_t n_configs = static_cast<std::size_t>(n_eps) * n_min_pts;

  const std::size_t est_mem_per_row =
    est_mem_per_row_multi<Index_>(N, n_eps, use_cuvs, neigh_hint);
  const std::size_t est_mem_fixed   = est_mem_fixed_multi<Index_>(N, n_eps, n_min_pts);

  ASSERT(est_mem_per_row > 0, "Estimated memory per row is 0");
  ASSERT(max_bytes_per_batch > est_mem_fixed,
         "Memória insuficiente para os buffers fixos: são necessários mais de %zu bytes "
         "para N=%d e %zu configurações",
         est_mem_fixed,
         (int)N,
         n_configs);

  std::size_t batch_size = (max_bytes_per_batch - est_mem_fixed) / est_mem_per_row;
  batch_size             = std::min(static_cast<std::size_t>(N), batch_size);
  if (batch_size < 1) batch_size = 1;

  // Mesma restrição de overflow do cuML: o CSR do lote indexa N * batch_size elementos.
  const Index_ MAX_LABEL = std::numeric_limits<Index_>::max();
  const std::size_t max_batch_by_index =
    static_cast<std::size_t>((MAX_LABEL - static_cast<Index_>(1)) / N);
  ASSERT(max_batch_by_index >= 1,
         "N=%d não permite nem um lote de uma linha com o tipo de índice escolhido",
         (int)N);
  if (batch_size > max_batch_by_index) {
    batch_size = max_batch_by_index;
    CUML_LOG_INFO("Batch size limited by the index type to %zu", batch_size);
  }

  estimated_memory = batch_size * est_mem_per_row + est_mem_fixed;
  return batch_size;
}

/**
 * Buffers dimensionados por nnz — o tamanho só se conhece depois da busca, então eles não
 * cabem no workspace, que é dimensionado antes.
 *
 * Existem por fora de `run_multi_grid_*` para poderem sobreviver entre chamadas. Sendo
 * locais da função, eram alocados e liberados a cada `fit_multi`: em heterogeneous_blobs_64d
 * são 9,4 GB de adj_graph, 9,4 GB de adj_graph_f e 1,2 GB de codes — 20 GB de cudaMalloc e
 * cudaFree por chamada, dentro da região cronometrada. É o mesmo defeito que o workspace
 * tinha, um nível abaixo, e maior: o workspace ali são 3,8 GB.
 *
 * O RMM deste build usa `cuda_memory_resource`, não pool, então cada alocação é um
 * cudaMalloc cru que sincroniza o dispositivo.
 *
 * Quem chama `fit_multi` em laço — warmup, repeat, uma grade por vez — deve passar um destes.
 */
template <typename Index_>
struct BuffersCsr {
  rmm::device_uvector<Index_> adj_graph;
  rmm::device_uvector<Index_> adj_graph_f;
  rmm::device_uvector<std::uint8_t> codes;

  explicit BuffersCsr(cudaStream_t stream)
    : adj_graph(0, stream), adj_graph_f(0, stream), codes(0, stream)
  {
  }
};

/**
 * Executa o DBSCAN para a grade de k valores de ε por l valores de minPts.
 *
 * @param[in]  handle       raft handle
 * @param[in]  x            pontos, N x D row-major, no device
 * @param[in]  N            número de pontos
 * @param[in]  D            dimensionalidade
 * @param[in]  eps2         k raios AO QUADRADO no device, em ordem CRESCENTE
 * @param[in]  n_eps        k
 * @param[in]  min_pts      l valores de minPts no device, em ordem CRESCENTE
 * @param[in]  n_min_pts    l
 * @param[out] labels       k*l * N rótulos, config-major, config = e * l + m
 * @param[in]  workspace    buffer temporário; se NULL, retorna o tamanho necessário
 * @param[in]  batch_size   linhas por lote
 * @param[in]  stream       CUDA stream
 * @param[out] neigh_medido se não NULL e o CSR do lote não couber, recebe o grau médio
 *                          medido e a execução aborta ANTES de alocar — cabe ao chamador
 *                          refazer o lote com esse valor (ver fit_multi)
 * @return tamanho do workspace quando workspace == NULL; 0 quando executou; e também 0
 *         quando abortou, caso em que *neigh_medido > 0
 */
template <typename Type_f, typename Index_ = int>
std::size_t run_multi_grid_codes(const raft::handle_t& handle,
                                 const Type_f* x,
                                 Index_ N,
                                 Index_ D,
                                 const Type_f* eps2,
                                 int n_eps,
                                 const Index_* min_pts,
                                 int n_min_pts,
                                 Index_* labels,
                                 void* workspace,
                                 std::size_t batch_size,
                                 cudaStream_t stream,
                                 Index_* neigh_medido      = nullptr,
                                 BuffersCsr<Index_>* buffers = nullptr)
{
  const std::size_t align     = 256;
  const std::size_t n_configs = static_cast<std::size_t>(n_eps) * n_min_pts;
  const Index_ n_batches =
    static_cast<Index_>((static_cast<std::size_t>(N) + batch_size - 1) / batch_size);
  const Index_ vd_stride = static_cast<Index_>(batch_size + 1);

  const std::size_t codes_size = raft::alignTo<std::size_t>(
    sizeof(std::uint8_t) * static_cast<std::size_t>(N) * batch_size, align);
  const std::size_t adj_size =
    raft::alignTo<std::size_t>(sizeof(bool) * static_cast<std::size_t>(N) * batch_size, align);
  const std::size_t core_pts_size =
    raft::alignTo<std::size_t>(sizeof(bool) * static_cast<std::size_t>(N) * n_configs, align);
  const std::size_t m_size = raft::alignTo<std::size_t>(sizeof(bool), align);
  const std::size_t vd_size =
    raft::alignTo<std::size_t>(sizeof(Index_) * static_cast<std::size_t>(vd_stride) * n_eps, align);
  const std::size_t ex_scan_size =
    raft::alignTo<std::size_t>(sizeof(Index_) * (batch_size + 1), align);
  const std::size_t row_cnt_size = raft::alignTo<std::size_t>(sizeof(Index_) * batch_size, align);
  const std::size_t labels_size =
    raft::alignTo<std::size_t>(sizeof(Index_) * static_cast<std::size_t>(N), align);

  const Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  ASSERT(static_cast<std::size_t>(N) * batch_size < static_cast<std::size_t>(MAX_LABEL),
         "An overflow occurred with the current choice of precision and the number of samples. "
         "(Max allowed batch size is %ld, but was %ld).",
         (unsigned long)((MAX_LABEL - static_cast<Index_>(1)) / N),
         (unsigned long)batch_size);

  if (workspace == NULL) {
    return codes_size + adj_size + core_pts_size + m_size + vd_size + ex_scan_size + row_cnt_size +
           2 * labels_size;
  }

  char* temp          = (char*)workspace;
  std::uint8_t* codes = (std::uint8_t*)temp;
  temp += codes_size;
  bool* adj = (bool*)temp;
  temp += adj_size;
  bool* core_pts = (bool*)temp;
  temp += core_pts_size;
  bool* m = (bool*)temp;
  temp += m_size;
  Index_* vd = (Index_*)temp;
  temp += vd_size;
  Index_* ex_scan = (Index_*)temp;
  temp += ex_scan_size;
  Index_* row_counters = (Index_*)temp;
  temp += row_cnt_size;
  Index_* labels_temp = (Index_*)temp;
  temp += labels_size;
  Index_* work_buffer = (Index_*)temp;
  temp += labels_size;

  BuffersCsr<Index_> buffers_locais(stream);
  auto& adj_graph = (buffers != nullptr ? *buffers : buffers_locais).adj_graph;
  // nnz por (lote, ε), em layout lote-major
  std::vector<Index_> batchadjlen(static_cast<std::size_t>(n_batches) * n_eps, 0);

  // Passagem 1: códigos de ε, graus e máscaras de pontos centrais.
  // Ordem reversa para deixar o lote 0 residente em memória e não recomputá-lo na
  // passagem 2 (mesma estratégia do cuML).
  for (int i = n_batches - 1; i >= 0; i--) {
    const Index_ start_vertex_id = static_cast<Index_>(i * batch_size);
    const Index_ n_points =
      std::min(static_cast<Index_>(N - i * batch_size), static_cast<Index_>(batch_size));

    CUML_LOG_DEBUG("- VertexDeg batch %d / %d (%d pontos)", i + 1, (int)n_batches, (int)n_points);

    VertexDeg::run_multi<Type_f, Index_>(
      x, N, D, start_vertex_id, n_points, eps2, n_eps, codes, vd, vd_stride, stream);

    for (int e = 0; e < n_eps; ++e) {
      raft::update_host(
        &batchadjlen[static_cast<std::size_t>(i) * n_eps + e], vd + e * vd_stride + n_points, 1, stream);
    }
    handle.sync_stream(stream);

    // Uma varredura dos graus alimenta as k*l máscaras.
    CorePoints::compute_grid<Index_, Index_>(handle,
                                             vd,
                                             vd_stride,
                                             n_eps,
                                             min_pts,
                                             n_min_pts,
                                             core_pts,
                                             N,
                                             start_vertex_id,
                                             n_points,
                                             stream);
  }

  raft::sparse::WeakCCState state(m);

  const Index_ maxadjlen = *std::max_element(batchadjlen.begin(), batchadjlen.end());
  last_execution_stats().max_nnz = static_cast<long long>(maxadjlen);
  for (Index_ batch = 0; batch < n_batches; ++batch) {
    last_execution_stats().total_nnz_max_eps += static_cast<long long>(
      batchadjlen[static_cast<std::size_t>(batch) * n_eps + (n_eps - 1)]);
  }
  CUML_LOG_DEBUG("Alocando %ld elementos para o grafo de adjacência", (unsigned long)maxadjlen);

  if (neigh_medido != nullptr &&
      !csr_do_lote_cabe<Index_>(maxadjlen, csr_bytes_por_entrada<Index_>(n_eps, false))) {
    *neigh_medido = grau_medido<Index_>(maxadjlen, static_cast<Index_>(batch_size));
    return 0;
  }

  adj_graph.resize(maxadjlen, stream);

  // Passagem 2: CSR (uma vez por lote e por ε) e rotulagem (uma vez por configuração).
  for (int i = 0; i < n_batches; i++) {
    const Index_ start_vertex_id = static_cast<Index_>(i * batch_size);
    const Index_ n_points =
      std::min(static_cast<Index_>(N - i * batch_size), static_cast<Index_>(batch_size));
    if (n_points <= 0) break;

    // i == 0 -> códigos e graus do lote 0 já estão em memória, vindos da passagem 1.
    if (i > 0) {
      VertexDeg::run_multi<Type_f, Index_>(
        x, N, D, start_vertex_id, n_points, eps2, n_eps, codes, vd, vd_stride, stream);
    }

    handle.sync_stream(stream);

    for (int e = 0; e < n_eps; ++e) {
      const Index_ nnz = batchadjlen[static_cast<std::size_t>(i) * n_eps + e];
      CUML_LOG_DEBUG("- AdjGraph batch %d eps %d com %ld nnz", i + 1, e, (unsigned long)nnz);

      // Adjacência do raio ε_e a partir dos códigos: nenhuma distância é recalculada.
      VertexDeg::codes_to_adj(
        codes, adj, static_cast<std::size_t>(n_points) * static_cast<std::size_t>(N), e, stream);

      AdjGraph::run<Index_>(handle,
                            adj,
                            vd + e * vd_stride,
                            adj_graph.data(),
                            nnz,
                            ex_scan,
                            N,
                            /* algo */ 1,
                            n_points,
                            row_counters,
                            stream);

      for (int mp = 0; mp < n_min_pts; ++mp) {
        const std::size_t config = static_cast<std::size_t>(e) * n_min_pts + mp;
        const bool* mask   = core_pts + config * static_cast<std::size_t>(N);
        Index_* out        = labels + config * static_cast<std::size_t>(N);

        raft::sparse::weak_cc_batched<Index_>(
          i == 0 ? out : labels_temp,
          ex_scan,
          adj_graph.data(),
          nnz,
          N,
          start_vertex_id,
          n_points,
          &state,
          stream,
          [mask, N] __device__(Index_ global_id) -> bool {
            return global_id < N
                     ? static_cast<bool>(__ldg(reinterpret_cast<const char*>(mask) + global_id))
                     : false;
          });

        if (i > 0) {
          // Os rótulos do lote atual precisam ser fundidos com os dos lotes anteriores;
          // usar o rótulo anterior como valor inicial do weak_cc daria resultado errado
          // (ver rapidsai/cuml#3094).
          MergeLabels::run<Index_>(handle, out, labels_temp, mask, work_buffer, m, N, stream);
        }
      }
    }
  }

  // Relabel final, por configuração, com as primitivas RAFT/Thrust do cuML.
  for (std::size_t config = 0; config < n_configs; ++config) {
    Index_* out = labels + config * static_cast<std::size_t>(N);
    CumlStages::relabel_for_sklearn<Index_>(handle, out, N, stream);
  }

  CUML_LOG_DEBUG("Done.");
  return 0;
}

#ifdef DBSCANMULTI_USE_CUVS

/**
 * Mesma grade, pelo caminho do cuML: a busca de vizinhança continua sendo a chamada ao
 * cuVS que o DBSCAN do cuML faz, e o multi-ε entra sobre o CSR que o AdjGraph do cuML
 * constrói.
 *
 * Por lote:
 *   1. cuVS epsilon_neighborhood::compute no MAIOR raio  -> adjacência densa + graus;
 *   2. AdjGraph::run do cuML                             -> CSR do lote no maior raio;
 *   3. anotação do CSR                                   -> menor ε de cada entrada e, de
 *      graça, os graus em todos os raios (O(nnz · D), não O(N² · D));
 *   4. por ε, o CSR sai por compactação das entradas anotadas (O(nnz), sem distância);
 *   5. por configuração, weak_cc_batched + MergeLabels do cuML, intocados.
 *
 * Com k = 1 os passos 3 e 4 não existem e o caminho é, kernel a kernel, o do cuML.
 *
 * @param[in] eps_host k raios originais no host, em ordem CRESCENTE; o VertexDeg do
 *                     cuML eleva cada raio ao quadrado internamente
 */
template <typename Type_f, typename Index_ = int>
std::size_t run_multi_grid_cuvs(const raft::handle_t& handle,
                                const Type_f* x,
                                Index_ N,
                                Index_ D,
                                const Type_f* eps_host,
                                const Type_f* eps2,
                                int n_eps,
                                const Index_* min_pts,
                                int n_min_pts,
                                Index_* labels,
                                void* workspace,
                                std::size_t batch_size,
                                cudaStream_t stream,
                                Index_* neigh_medido        = nullptr,
                                PerfilMulti* perfil         = nullptr,
                                BuffersCsr<Index_>* buffers = nullptr)
{
  const std::size_t align     = 256;
  const std::size_t n_configs = static_cast<std::size_t>(n_eps) * n_min_pts;
  const Index_ n_batches =
    static_cast<Index_>((static_cast<std::size_t>(N) + batch_size - 1) / batch_size);
  const Index_ vd_stride = static_cast<Index_>(batch_size + 1);
  const bool multi_eps   = n_eps > 1;

  const std::size_t adj_size =
    raft::alignTo<std::size_t>(sizeof(bool) * static_cast<std::size_t>(N) * batch_size, align);
  const std::size_t core_pts_size =
    raft::alignTo<std::size_t>(sizeof(bool) * static_cast<std::size_t>(N) * n_configs, align);
  const std::size_t m_size = raft::alignTo<std::size_t>(sizeof(bool), align);
  const std::size_t vd_size =
    raft::alignTo<std::size_t>(sizeof(Index_) * static_cast<std::size_t>(vd_stride) * n_eps, align);
  const std::size_t ex_scan_size =
    raft::alignTo<std::size_t>(sizeof(Index_) * (batch_size + 1), align);
  const std::size_t row_cnt_size = raft::alignTo<std::size_t>(sizeof(Index_) * batch_size, align);
  const std::size_t labels_size =
    raft::alignTo<std::size_t>(sizeof(Index_) * static_cast<std::size_t>(N), align);

  const Index_ MAX_LABEL = std::numeric_limits<Index_>::max();

  ASSERT(static_cast<std::size_t>(N) * batch_size < static_cast<std::size_t>(MAX_LABEL),
         "An overflow occurred with the current choice of precision and the number of samples. "
         "(Max allowed batch size is %ld, but was %ld).",
         (unsigned long)((MAX_LABEL - static_cast<Index_>(1)) / N),
         (unsigned long)batch_size);

  if (workspace == NULL) {
    return adj_size + core_pts_size + m_size + vd_size + (multi_eps ? 2 : 1) * ex_scan_size +
           row_cnt_size + 2 * labels_size;
  }

  char* temp = (char*)workspace;
  bool* adj  = (bool*)temp;
  temp += adj_size;
  bool* core_pts = (bool*)temp;
  temp += core_pts_size;
  bool* m = (bool*)temp;
  temp += m_size;
  Index_* vd = (Index_*)temp;
  temp += vd_size;
  Index_* ex_scan = (Index_*)temp;
  temp += ex_scan_size;
  Index_* ex_scan_f = nullptr;
  if (multi_eps) {
    ex_scan_f = (Index_*)temp;
    temp += ex_scan_size;
  }
  Index_* row_counters = (Index_*)temp;
  temp += row_cnt_size;
  Index_* labels_temp = (Index_*)temp;
  temp += labels_size;
  Index_* work_buffer = (Index_*)temp;
  temp += labels_size;

  // Graus do maior raio: é o que o cuVS preenche e o que o AdjGraph consome.
  Index_* vd_max = vd + static_cast<std::size_t>(n_eps - 1) * vd_stride;

  // VertexDeg::run recebe o raio original e faz eps*eps internamente. A cópia em device ao
  // quadrado continua servindo à anotação, que compara os k valores dentro do kernel.
  const Type_f eps_max = eps_host[static_cast<std::size_t>(n_eps) - 1];

  // Buffers dimensionados por nnz, que só se conhece depois da busca — mesma estratégia do
  // cuML, que mantém adj_graph fora do workspace e o redimensiona. Reaproveitados entre
  // chamadas quando o chamador passa um BuffersCsr; ver o comentário daquela struct.
  BuffersCsr<Index_> buffers_locais(stream);
  BuffersCsr<Index_>& buf = buffers != nullptr ? *buffers : buffers_locais;

  auto& adj_graph   = buf.adj_graph;
  auto& adj_graph_f = buf.adj_graph_f;
  auto& codes       = buf.codes;

  std::vector<Index_> batchadjlen(static_cast<std::size_t>(n_batches), 0);
  std::vector<Index_> nnz_eps(static_cast<std::size_t>(n_batches) * n_eps, 0);

  // Rota de cada lote: 1 = anotar o CSR e compactar (bom em grafo esparso), 0 = refazer a
  // busca uma vez por raio (bom em grafo denso). Por lote, e não por dataset: em dados
  // heterogêneos a densidade varia entre lotes.
  std::vector<char> anota(static_cast<std::size_t>(n_batches), 1);

  // Cresce sob demanda, e nunca com `resize`.
  //
  // `resize` preserva o conteúdo: aloca o buffer novo, COPIA o antigo e só então o libera.
  // Aqui o conteúdo é sempre reescrito do zero, então a cópia é pura perda — até 9 GB de
  // device-to-device por crescimento em dados densos — e os dois buffers coexistem, o que
  // dobra o pico exatamente no regime em que a memória é mais apertada.
  auto ensure = [&](auto& buffer, Index_ needed) {
    using Buf = std::decay_t<decltype(buffer)>;
    if (static_cast<std::size_t>(buffer.size()) >= static_cast<std::size_t>(needed)) return;
    buffer = Buf(0, stream);  // libera antes de pedir o novo
    buffer = Buf(static_cast<std::size_t>(needed), stream);
  };

  // Passagem 1: graus em todos os raios e máscaras de pontos centrais. Ordem reversa para
  // deixar o lote 0 residente, como o cuML faz.
  for (int i = n_batches - 1; i >= 0; i--) {
    const Index_ start_vertex_id = static_cast<Index_>(i * batch_size);
    const Index_ n_points =
      std::min(static_cast<Index_>(N - i * batch_size), static_cast<Index_>(batch_size));

    CUML_LOG_DEBUG("- VertexDeg batch %d / %d (%d pontos)", i + 1, (int)n_batches, (int)n_points);

    {
      Cronometro t(DBM_FASE(perfil, busca_p1), stream);
      CumlStages::vertex_degree_l2<Type_f, Index_>(
        handle, x, N, D, start_vertex_id, n_points, eps_max, adj, vd_max, stream);
    }

    Index_ nnz = 0;
    raft::update_host(&nnz, vd_max + n_points, 1, stream);
    handle.sync_stream(stream);
    batchadjlen[static_cast<std::size_t>(i)] = nnz;
    last_execution_stats().max_nnz =
      std::max(last_execution_stats().max_nnz, static_cast<long long>(nnz));
    last_execution_stats().total_nnz_max_eps += static_cast<long long>(nnz);

    // Primeiro ponto em que o grau real é conhecido. Se o lote foi dimensionado por um
    // palpite otimista, é aqui que o CSR estouraria — abortar agora custa uma passagem de
    // busca, e estourar custa o job inteiro.
    if (neigh_medido != nullptr &&
        !csr_do_lote_cabe<Index_>(nnz, csr_bytes_por_entrada<Index_>(n_eps, true))) {
      *neigh_medido = grau_medido<Index_>(nnz, n_points);
      return 0;
    }

    // A rota se decide AQUI, com o nnz medido — antes da busca ninguém sabe se o grafo é
    // esparso.
    if (multi_eps) {
      anota[static_cast<std::size_t>(i)] =
        rota_forcada() != 0
          ? (rota_forcada() == 1)
          : anotar_compensa(static_cast<long long>(nnz),
                            static_cast<long long>(N) * n_points,
                            n_eps,
                            static_cast<long long>(D));
      if (anota[static_cast<std::size_t>(i)]) {
        ++last_execution_stats().annotated_batches;
        last_execution_stats().batch_routes[static_cast<std::size_t>(i)] =
          BatchRoute::Annotated;
      } else {
        ++last_execution_stats().dense_batches;
        last_execution_stats().batch_routes[static_cast<std::size_t>(i)] = BatchRoute::Dense;
      }
    }

    if (multi_eps && !anota[static_cast<std::size_t>(i)]) {
      // Rota densa: uma chamada ao cuVS por raio dá os k graus direto, sem CSR e sem
      // anotação. São k buscas em vez de uma, mas a busca é tiled e a anotação é gather
      // aleatório: no regime denso as k buscas saem ~7x mais baratas que uma anotação.
      // O maior raio já foi calculado acima e seus graus estão em vd_max.
      Cronometro t(DBM_FASE(perfil, busca_p1), stream);
      for (int e = 0; e < n_eps - 1; ++e) {
        CumlStages::vertex_degree_l2<Type_f, Index_>(
          handle,
          x,
          N,
          D,
          start_vertex_id,
          n_points,
          eps_host[static_cast<std::size_t>(e)],
          adj,
          vd + static_cast<std::size_t>(e) * vd_stride,
          stream);
      }
    } else if (multi_eps) {
      ensure(adj_graph, nnz);
      ensure(codes, nnz);

      {
        Cronometro t(DBM_FASE(perfil, csr_p1), stream);
        AdjGraph::run<Index_>(handle,
                              adj,
                              vd_max,
                              adj_graph.data(),
                              nnz,
                              ex_scan,
                              N,
                              /* algo */ 1,
                              n_points,
                              row_counters,
                              stream);
      }

      {
        Cronometro t(DBM_FASE(perfil, anotacao_p1), stream);
        CsrEps::annotate<Type_f, Index_>(x,
                                         D,
                                         start_vertex_id,
                                         n_points,
                                         ex_scan,
                                         vd_max,
                                         adj_graph.data(),
                                         eps2,
                                         n_eps,
                                         codes.data(),
                                         vd,
                                         vd_stride,
                                         stream);
      }

    }

    if (multi_eps) {
      for (int e = 0; e < n_eps; ++e) {
        raft::update_host(&nnz_eps[static_cast<std::size_t>(i) * n_eps + e],
                          vd + static_cast<std::size_t>(e) * vd_stride + n_points,
                          1,
                          stream);
      }
      handle.sync_stream(stream);
    } else {
      nnz_eps[static_cast<std::size_t>(i)] = nnz;
    }

    {
      Cronometro t(DBM_FASE(perfil, nucleo), stream);
      CorePoints::compute_grid<Index_, Index_>(handle,
                                               vd,
                                               vd_stride,
                                               n_eps,
                                               min_pts,
                                               n_min_pts,
                                               core_pts,
                                               N,
                                               start_vertex_id,
                                               n_points,
                                               stream);
    }
  }

  raft::sparse::WeakCCState state(m);

  const Index_ maxadjlen = *std::max_element(batchadjlen.begin(), batchadjlen.end());
  CUML_LOG_DEBUG("Alocando %ld elementos para o grafo de adjacencia", (unsigned long)maxadjlen);
  ensure(adj_graph, maxadjlen);

  // codes e adj_graph_f só servem à rota que anota. Se nenhum lote anota, são dezenas de GB
  // que não precisam existir — e é justamente no caso denso, onde a memória é mais
  // apertada, que nenhum lote anota.
  Index_ maxadjlen_anotado = 0;
  for (std::size_t i = 0; i < anota.size(); ++i) {
    if (multi_eps && anota[i]) {
      maxadjlen_anotado = std::max(maxadjlen_anotado, batchadjlen[i]);
    }
  }
  const bool algum_anota = maxadjlen_anotado > 0;
  if (algum_anota) {
    // Um lote denso nunca usa estes buffers. Dimensioná-los pelo maior CSR global faria
    // um único lote denso eliminar a economia de memória da decisão adaptativa por lote.
    ensure(codes, maxadjlen_anotado);
    ensure(adj_graph_f, maxadjlen_anotado);
  }

  // Rotula todas as configurações de um raio, sobre um CSR já pronto. Fatorado porque as
  // duas rotas chegam aqui com CSRs de origens diferentes e a rotulagem é idêntica.
  auto rotular = [&](int e,
                     int i,
                     const Index_* csr_offsets,
                     const Index_* csr_columns,
                     Index_ nnz,
                     Index_ start_vertex_id,
                     Index_ n_points) {
    for (int mp = 0; mp < n_min_pts; ++mp) {
      const std::size_t config = static_cast<std::size_t>(e) * n_min_pts + mp;
      const bool* mask         = core_pts + config * static_cast<std::size_t>(N);
      Index_* out              = labels + config * static_cast<std::size_t>(N);

      // Cronometrado num local e só depois somado nos dois campos: dois Cronometro
      // aninhados sincronizariam o stream duas vezes e cobrariam a espera de um no outro.
      double ms_wcc = 0;
      {
        Cronometro t(perfil ? &ms_wcc : nullptr, stream);
        raft::sparse::weak_cc_batched<Index_>(
          i == 0 ? out : labels_temp,
          csr_offsets,
          csr_columns,
          nnz,
          N,
          start_vertex_id,
          n_points,
          &state,
          stream,
          [mask, N] __device__(Index_ global_id) -> bool {
            return global_id < N
                     ? static_cast<bool>(__ldg(reinterpret_cast<const char*>(mask) + global_id))
                     : false;
          });
      }
      if (perfil != nullptr && e < PerfilMulti::kMaxEpsPerfil) {
        perfil->rotulagem += ms_wcc;
        perfil->rotulagem_eps[e] += ms_wcc;
        perfil->nnz_rotulado[e] += static_cast<long long>(nnz);
        perfil->chamadas_eps[e] += 1;
      }

      if (i > 0) {
        // Os rótulos deste lote são fundidos com os dos anteriores; usar o rótulo anterior
        // como valor inicial do weak_cc daria resultado errado (rapidsai/cuml#3094).
        Cronometro t(DBM_FASE(perfil, fusao), stream);
        MergeLabels::run<Index_>(handle, out, labels_temp, mask, work_buffer, m, N, stream);
      }
    }
  };

  // A passagem 1 termina no lote 0 (ela corre em ordem reversa), então a adjacência dele
  // ainda está em memória — mas só serve se aquele lote tiver anotado: a rota densa deixa
  // `adj` no penúltimo raio, não no maior.
  const bool adj_do_lote0_vale = multi_eps ? (anota[0] != 0) : true;

  // Passagem 2: CSR por (lote, ε) e rotulagem por configuração.
  for (int i = 0; i < n_batches; i++) {
    const Index_ start_vertex_id = static_cast<Index_>(i * batch_size);
    const Index_ n_points =
      std::min(static_cast<Index_>(N - i * batch_size), static_cast<Index_>(batch_size));
    if (n_points <= 0) break;

    const Index_ nnz_max = batchadjlen[static_cast<std::size_t>(i)];

    if (multi_eps && !anota[static_cast<std::size_t>(i)]) {
      // ---- rota densa: um CSR por raio, direto do cuVS, sem anotação nem compactação ---
      // Custa k buscas e k AdjGraph. No regime denso sai ~3,5x mais barato que anotar, e
      // usa só kernels do cuML: nenhuma distância é buscada esparsamente.
      for (int e = 0; e < n_eps; ++e) {
        Index_* vd_e = vd + static_cast<std::size_t>(e) * vd_stride;

        {
          Cronometro t(DBM_FASE(perfil, busca_p2), stream);
          CumlStages::vertex_degree_l2<Type_f, Index_>(handle,
                                                       x,
                                                       N,
                                                       D,
                                                       start_vertex_id,
                                                       n_points,
                                                       eps_host[static_cast<std::size_t>(e)],
                                                       adj,
                                                       vd_e,
                                                       stream);
        }

        // Sem sync: nada é lido para o host aqui. `nnz_e` vem de nnz_eps, preenchido na
        // passagem 1. Sincronizar impediria a busca do próximo raio de se sobrepor ao
        // AdjGraph e à rotulagem deste.
        const Index_ nnz_e = nnz_eps[static_cast<std::size_t>(i) * n_eps + e];

        {
          Cronometro t(DBM_FASE(perfil, csr_p2), stream);
          AdjGraph::run<Index_>(handle,
                                adj,
                                vd_e,
                                adj_graph.data(),
                                nnz_e,
                                ex_scan,
                                N,
                                /* algo */ 1,
                                n_points,
                                row_counters,
                                stream);
        }

        CUML_LOG_DEBUG(
          "- batch %d eps %d com %ld nnz (rota densa)", i + 1, e, (unsigned long)nnz_e);
        rotular(e, i, ex_scan, adj_graph.data(), nnz_e, start_vertex_id, n_points);
      }
      continue;
    }

    // ---- rota esparsa: um CSR no maior raio, anotado, e os menores por compactação -----
    if (i > 0 || !adj_do_lote0_vale) {
      Cronometro t(DBM_FASE(perfil, busca_p2), stream);
      CumlStages::vertex_degree_l2<Type_f, Index_>(
        handle, x, N, D, start_vertex_id, n_points, eps_max, adj, vd_max, stream);
    }

    // Sem sync: `nnz_max` veio de batchadjlen, lido na passagem 1.
    {
      Cronometro t(DBM_FASE(perfil, csr_p2), stream);
      AdjGraph::run<Index_>(handle,
                            adj,
                            vd_max,
                            adj_graph.data(),
                            nnz_max,
                            ex_scan,
                            N,
                            /* algo */ 1,
                            n_points,
                            row_counters,
                            stream);
    }

    // A anotação é refeita aqui mesmo no lote 0: o adj_to_csr do RAFT insere as colunas de
    // cada linha por atômico, então a ordem dentro da linha pode diferir da passagem 1 e os
    // códigos daquela passagem não corresponderiam a estas entradas.
    if (multi_eps) {
      Cronometro t(DBM_FASE(perfil, anotacao_p2), stream);
      CsrEps::annotate<Type_f, Index_>(x,
                                       D,
                                       start_vertex_id,
                                       n_points,
                                       ex_scan,
                                       vd_max,
                                       adj_graph.data(),
                                       eps2,
                                       n_eps,
                                       codes.data(),
                                       vd,
                                       vd_stride,
                                       stream);
    }

    for (int e = 0; e < n_eps; ++e) {
      const Index_* csr_offsets = ex_scan;
      const Index_* csr_columns = adj_graph.data();
      Index_ nnz                = nnz_max;

      // O maior raio já é o CSR construído; os menores saem por compactação.
      if (multi_eps && e < n_eps - 1) {
        Cronometro t(DBM_FASE(perfil, filtro), stream);
        CsrEps::filter<Index_>(handle,
                               ex_scan,
                               vd_max,
                               adj_graph.data(),
                               codes.data(),
                               n_points,
                               e,
                               vd + static_cast<std::size_t>(e) * vd_stride,
                               ex_scan_f,
                               adj_graph_f.data(),
                               stream);
        csr_offsets = ex_scan_f;
        csr_columns = adj_graph_f.data();
        nnz         = nnz_eps[static_cast<std::size_t>(i) * n_eps + e];
      }

      CUML_LOG_DEBUG("- batch %d eps %d com %ld nnz", i + 1, e, (unsigned long)nnz);
      rotular(e, i, csr_offsets, csr_columns, nnz, start_vertex_id, n_points);
    }
  }

  {
    Cronometro t(DBM_FASE(perfil, relabel), stream);
    for (std::size_t config = 0; config < n_configs; ++config) {
      Index_* out = labels + config * static_cast<std::size_t>(N);
      CumlStages::relabel_for_sklearn<Index_>(handle, out, N, stream);
    }
  }

  if (perfil != nullptr) {
    perfil->n_lotes = static_cast<int>(n_batches);
    perfil->nnz_max = static_cast<long long>(maxadjlen);
    perfil->execucoes += 1;
    for (int b = 0; b < n_batches; ++b) {
      if (anota[static_cast<std::size_t>(b)]) {
        perfil->lotes_anotados += 1;
      } else {
        perfil->lotes_densos += 1;
      }
    }
  }

  CUML_LOG_DEBUG("Done.");
  return 0;
}

#endif  // DBSCANMULTI_USE_CUVS

/** Qual busca de vizinhança usar. Ver o cabeçalho deste arquivo. */
enum class Backend { Cuvs, Codes };

inline bool backend_uses_cuvs(Backend backend)
{
#ifdef DBSCANMULTI_USE_CUVS
  return backend == Backend::Cuvs;
#else
  (void)backend;
  return false;
#endif
}

/**
 * Driver equivalente ao dbscanFitImpl do cuML: estima o lote, aloca o workspace e executa.
 *
 * @param[in] max_bytes_per_batch  orçamento de memória; 0 = estimar a partir da GPU
 * @param[in] backend              busca de vizinhança: cuVS (padrão, é a do cuML) ou o
 *                                 kernel de códigos
 * @param[in] eps_host             raios originais no host; o VertexDeg do cuML calcula
 *                                 eps² internamente. O backend codes usa `eps2` no device.
 * @param[in] neigh_per_row        vizinhos esperados por linha para dimensionar o lote;
 *                                 0 = pior caso N, o padrão do cuML. É um palpite sobre os
 *                                 dados: se o grau real for maior, o lote sai grande demais
 *                                 e o CSR não cabe. Nesse caso o runner mede o grau e o
 *                                 lote é refeito automaticamente — custa uma passagem de
 *                                 busca, não o processo
 * @param[in,out] workspace_externo  buffer reaproveitado entre chamadas. Se NULL, um é
 *                                 alocado e liberado a cada chamada — cômodo, mas o
 *                                 workspace chega a dezenas de GB (a adjacência densa é
 *                                 N x batch_size bytes) e alocar/liberar isso a cada
 *                                 chamada custa caro: medimos a segunda chamada em diante
 *                                 custando 2,3x a primeira. Quem chama em laço deve passar
 *                                 um buffer aqui.
 */
template <typename Type_f, typename Index_ = int>
void fit_multi(const raft::handle_t& handle,
               const Type_f* x,
               Index_ N,
               Index_ D,
               const Type_f* eps_host,
               const Type_f* eps2,
               int n_eps,
               const Index_* min_pts,
               int n_min_pts,
               Index_* labels,
               std::size_t max_bytes_per_batch,
               cudaStream_t stream,
               Backend backend                          = Backend::Cuvs,
               Index_ neigh_per_row                     = 0,
               rmm::device_uvector<char>* workspace_externo = nullptr,
               PerfilMulti* perfil                      = nullptr,
               BuffersCsr<Index_>* buffers_externos     = nullptr)
{
  ASSERT(N > 0, "No rows in the input array. DBSCAN cannot be fitted!");
  ASSERT(n_eps > 0, "At least one eps value is required");
  ASSERT(n_min_pts > 0, "At least one minPts value is required");

#ifndef DBSCANMULTI_USE_CUVS
  ASSERT(backend != Backend::Cuvs,
         "Binário compilado sem cuVS; recompile com BACKEND=cuvs ou use --backend codes");
#endif

  const bool use_cuvs = backend_uses_cuvs(backend);
  ExecutionStats& stats = last_execution_stats();
  stats.reset_preserving_capacity();

  if (max_bytes_per_batch == 0) {
    std::size_t free_memory, total_memory;
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
    // A consulta ocorre depois de o dataset estar residente. Usar memória total podia
    // prometer bytes ocupados por outros processos do nó e só descobrir no allocator.
    // Por outro lado, nas repetições o nosso próprio workspace externo já está alocado:
    // ele some de `free_memory`, mas é integralmente reutilizável nesta chamada. Somá-lo
    // de volta mantém o lote estável entre warmup/repeat sem contar memória alheia.
    const std::size_t reusable_workspace =
      workspace_externo != nullptr ? workspace_externo->size() : 0;
    const std::size_t available =
      reusable_workspace > total_memory - std::min(total_memory, free_memory)
        ? total_memory
        : std::min(total_memory, free_memory + reusable_workspace);
    max_bytes_per_batch = 80 * available / 100;
  }
  stats.max_bytes_per_batch = max_bytes_per_batch;

  // `neigh_per_row` é um palpite sobre os dados, e um palpite errado só se revela depois da
  // busca. Quando o CSR medido não couber, o runner devolve o grau real em vez de estourar,
  // e refazemos com um teto de lote calculado desse grau. Três tentativas bastam: a
  // primeira usa o palpite, a segunda o grau medido, e a terceira cobre o caso de um lote
  // posterior ser mais denso que o primeiro que estourou.
  std::size_t teto_lote = static_cast<std::size_t>(N);

  for (int tentativa = 0; tentativa < 3; ++tentativa) {
    std::size_t estimated_memory = 0;
    const std::size_t batch_size =
      std::min(compute_batch_size_multi<Index_>(
                 estimated_memory, N, n_eps, n_min_pts, max_bytes_per_batch, use_cuvs, neigh_per_row),
               teto_lote);

    stats.attempts          = tentativa + 1;
    stats.batch_size        = batch_size;
    stats.batches           = static_cast<int>((static_cast<std::size_t>(N) + batch_size - 1) /
                                     batch_size);
    stats.dense_batches     = 0;
    stats.annotated_batches = 0;
    stats.batch_routes.clear();
    stats.batch_routes.resize(static_cast<std::size_t>(stats.batches),
                              BatchRoute::NotApplicable);
    stats.max_nnz           = 0;
    stats.total_nnz_max_eps = 0;

    CUML_LOG_DEBUG(
      "Batch size %zu, memória estimada %.2f MB", batch_size, (double)estimated_memory * 1e-6);

    Index_ medido = 0;

    auto run = [&](void* workspace) -> std::size_t {
#ifdef DBSCANMULTI_USE_CUVS
      if (use_cuvs) {
        return run_multi_grid_cuvs<Type_f, Index_>(handle,
                                                   x,
                                                   N,
                                                   D,
                                                   eps_host,
                                                   eps2,
                                                   n_eps,
                                                   min_pts,
                                                   n_min_pts,
                                                   labels,
                                                   workspace,
                                                   batch_size,
                                                   stream,
                                                   workspace == NULL ? nullptr : &medido,
                                                   workspace == NULL ? nullptr : perfil,
                                                   buffers_externos);
      }
#endif
      return run_multi_grid_codes<Type_f, Index_>(handle,
                                                  x,
                                                  N,
                                                  D,
                                                  eps2,
                                                  n_eps,
                                                  min_pts,
                                                  n_min_pts,
                                                  labels,
                                                  workspace,
                                                  batch_size,
                                                  stream,
                                                  workspace == NULL ? nullptr : &medido,
                                                  buffers_externos);
    };

    const std::size_t workspace_size = run(NULL);

    if (workspace_externo != nullptr) {
      // Cresce sob demanda e nunca encolhe. O orçamento automático pode contar o buffer
      // atual como memória reutilizável; ao crescer, ele precisa ser liberado ANTES da
      // nova alocação, pois `device_uvector(workspace_size)` no lado direito coexistiria
      // temporariamente com o antigo e poderia causar OOM apesar de o orçamento caber.
      if (workspace_externo->size() < workspace_size) {
        *workspace_externo = rmm::device_uvector<char>(0, stream);
        handle.sync_stream(stream);
        *workspace_externo = rmm::device_uvector<char>(workspace_size, stream);
      }
      run(workspace_externo->data());
    } else {
      rmm::device_uvector<char> workspace(workspace_size, stream);
      run(workspace.data());
    }

    if (medido == 0) return;  // executou

    ++csr_correcoes_de_lote();
    ++stats.batch_corrections;

    // Na primeira correção, o grau medido — barato e quase sempre suficiente. Se ainda
    // assim não couber, o lote seguinte era mais denso que o que estourou: aí vai o pior
    // caso (N vizinhos por linha), que cabe se algum lote couber.
    const Index_ grau_do_teto   = (tentativa == 0) ? medido : N;
    const std::size_t novo_teto = linhas_de_csr_que_cabem<Index_>(
      grau_do_teto, csr_bytes_por_entrada<Index_>(n_eps, use_cuvs));

    ASSERT(novo_teto < batch_size,
           "O CSR não cabe na memória nem com um lote de %zu linhas (N=%d, grau medido %ld, "
           "ou %ld%% de N). Reduza --max-mbytes-per-batch, ou reveja o ε: um raio em que "
           "cada ponto alcança essa fração dos outros já não separa grupo nenhum.",
           batch_size,
           (int)N,
           (unsigned long)medido,
           (unsigned long)(100.0 * medido / N));

    CUML_LOG_INFO("Grau real %ld por linha (palpite %ld); lote de %zu para %zu linhas",
                  (unsigned long)medido,
                  (unsigned long)(neigh_per_row > 0 ? neigh_per_row : N),
                  batch_size,
                  novo_teto);

    teto_lote = novo_teto;
  }

  ASSERT(false, "Não foi possível dimensionar o lote em 3 tentativas");
}

}  // namespace Multi
}  // namespace Dbscan
}  // namespace ML
