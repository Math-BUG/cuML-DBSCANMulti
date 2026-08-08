/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Universidade Federal de Viçosa (UFV)
 * SPDX-License-Identifier: Apache-2.0
 *
 * Executável CUDA do cuML-DBSCANMulti.
 *
 * Implementa o contrato de linha de comando do DBSCANMultiE, para reaproveitar o harness
 * de benchmark e validação já existente:
 *
 *   ./dbscan_multi --input points.f32 --output labels.i32 --n N --d D \
 *                  --eps E[,E2,...] --min-samples M[,M2,...] --json
 *   ./dbscan_multi ... --eps-min 0.1 --eps-max 0.5 --eps-step 0.1 ...
 *
 * - points.f32: matriz row-major float32, sem cabeçalho;
 * - labels.i32: um int32 por ponto e por configuração, em ordem config-major;
 * - a última linha do stdout é o JSON de resultado, com fit_ms e configuration_count.
 *
 * Extensão em relação ao contrato original: --min-samples também aceita lista. Com k
 * valores de ε e l de minPts saem k*l configurações, em ordem eps-major
 * (config = e * l + m), como reportado no campo "config_order" do JSON.
 *
 * fit_ms cobre a chamada completa do algoritmo, incluindo a alocação do workspace, e
 * exclui leitura de arquivo e as transferências de entrada e saída. Com --repeat R é a
 * mediana de R execuções; --warmup W descarta as W primeiras, que pagam o carregamento do
 * módulo CUDA.
 */

#include "multi/runner_multi.cuh"

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string input;
  std::string output;
  int n       = 0;
  int d       = 0;
  std::vector<float> eps;
  float eps_min                   = 0.0f;
  float eps_max                   = 0.0f;
  float eps_step                  = 0.0f;
  bool has_eps_range              = false;
  std::vector<int> min_samples;
  std::size_t max_bytes_per_batch = 0;
  int warmup                      = 0;
  int repeat                      = 1;
  std::string backend             = "cuvs";
  std::string index               = "auto";
  int neigh_per_row               = 0;
  bool solo                       = false;
  bool solo_isolado               = false;  // recria tudo por configuração (reprodutor)
  bool json                       = false;
  bool selftest                   = false;
  bool perfil                     = false;  // tempo por fase dentro do runner
};

/** Traduz o nome do backend, validando cedo para não falhar só na GPU. */
ML::Dbscan::Multi::Backend parse_backend(const std::string& nome)
{
  if (nome == "cuvs") return ML::Dbscan::Multi::Backend::Cuvs;
  if (nome == "codes") return ML::Dbscan::Multi::Backend::Codes;
  std::cerr << "erro: --backend deve ser 'cuvs' ou 'codes', recebi '" << nome << "'\n";
  std::exit(2);
}

void print_usage()
{
  std::cerr
    << "uso: dbscan_multi --input points.f32 --n N --d D\n"
       "                  (--eps E[,E2,...] | --eps-min A --eps-max B --eps-step S)\n"
       "                  --min-samples M[,M2,...]\n"
       "                  [--output labels.i32] [--max-mbytes-per-batch MB]\n"
       "                  [--backend cuvs|codes] [--index auto|int32|int64]\n"
       "                  [--warmup W] [--repeat R] [--json]\n"
       "     dbscan_multi --selftest [--backend cuvs|codes] [--json]\n"
       "\n"
       "  --backend cuvs (padrão) usa a mesma busca de vizinhança do DBSCAN do cuML e faz\n"
       "  o multi-eps sobre o CSR dele; codes usa o kernel próprio, sem libcuvs.\n"
       "  --index auto (padrão) usa int64 quando N*N estoura int32 (N >= 46341), porque\n"
       "  int32 trava o tamanho do lote em MAX_INT/N e cada lote a mais custa uma rodada\n"
       "  de fusão de rótulos POR CONFIGURAÇÃO.\n"
       "  --neigh-per-row V dimensiona o lote supondo V vizinhos por linha em vez do pior\n"
       "  caso N. É o parâmetro neigh_per_row do cuML, que o cuML não expõe. Reduz o número\n"
       "  de lotes; se V ficar muito abaixo do grau real, a alocação do CSR falha.\n"
       "  --eps e --min-samples aceitam listas separadas por vírgula. Com k valores de eps\n"
       "  e l de min-samples são avaliadas k*l configurações, em ordem eps-major\n"
       "  (config = e * l + m). Os valores efetivos e sua ordem vão no JSON.\n"
       "  A faixa --eps-min/--eps-max/--eps-step é mutuamente exclusiva com --eps.\n"
       "  Limite: no máximo 16 valores de eps.\n"
       "  --warmup W descarta W execuções antes de medir (a primeira paga o carregamento\n"
       "  do módulo CUDA); --repeat R mede R execuções e reporta a mediana em fit_ms, com\n"
       "  todos os tempos em fit_ms_all.\n"
       "  --solo mede também cada configuração ISOLADA (k=1, l=1), no mesmo processo,\n"
       "  compartilhando handle e workspace entre elas — que é como se deve chamar em laço.\n"
       "  Sai em solo_ms, e é o denominador do ganho do multi.\n"
       "  --solo-isolado recria tudo por configuração, inclusive o workspace. Existe para\n"
       "  reproduzir a degradação de 2,3x dos jobs 4862/4863; não use para medir.\n";
}

std::vector<int> parse_int_list(const std::string& raw)
{
  std::vector<int> values;
  std::stringstream ss(raw);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) values.push_back(std::stoi(item));
  }
  return values;
}

std::vector<float> parse_float_list(const std::string& raw)
{
  std::vector<float> values;
  std::stringstream ss(raw);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) values.push_back(std::stof(item));
  }
  return values;
}

/**
 * Expande a faixa de ε. O limite máximo entra quando pertence à progressão: uma faixa não
 * divisível como 0,1..0,55 com passo 0,2 produz 0,1 / 0,3 / 0,5 — mesma regra do
 * DBSCANMultiE, para que os arquivos de configuração do harness continuem valendo.
 */
std::vector<float> expand_eps_range(float eps_min, float eps_max, float eps_step)
{
  std::vector<float> values;
  if (eps_step <= 0.0f) {
    std::cerr << "erro: --eps-step deve ser positivo\n";
    std::exit(2);
  }
  if (eps_max < eps_min) {
    std::cerr << "erro: --eps-max deve ser maior ou igual a --eps-min\n";
    std::exit(2);
  }
  const double tolerance = 1e-9 * std::max(1.0, static_cast<double>(std::fabs(eps_max)));
  for (int i = 0;; ++i) {
    const double value = static_cast<double>(eps_min) + i * static_cast<double>(eps_step);
    if (value > static_cast<double>(eps_max) + tolerance) break;
    values.push_back(static_cast<float>(value));
    if (values.size() > static_cast<std::size_t>(ML::Dbscan::Multi::VertexDeg::kMaxEps)) break;
  }
  return values;
}

bool parse_args(int argc, char** argv, Options& opt)
{
  bool has_eps_list = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next             = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "erro: " << name << " exige um valor\n";
        std::exit(2);
      }
      return argv[++i];
    };

    if (arg == "--input") {
      opt.input = next("--input");
    } else if (arg == "--output") {
      opt.output = next("--output");
    } else if (arg == "--n") {
      opt.n = std::stoi(next("--n"));
    } else if (arg == "--d") {
      opt.d = std::stoi(next("--d"));
    } else if (arg == "--eps") {
      opt.eps      = parse_float_list(next("--eps"));
      has_eps_list = true;
    } else if (arg == "--eps-min") {
      opt.eps_min        = std::stof(next("--eps-min"));
      opt.has_eps_range  = true;
    } else if (arg == "--eps-max") {
      opt.eps_max       = std::stof(next("--eps-max"));
      opt.has_eps_range = true;
    } else if (arg == "--eps-step") {
      opt.eps_step      = std::stof(next("--eps-step"));
      opt.has_eps_range = true;
    } else if (arg == "--min-samples") {
      opt.min_samples = parse_int_list(next("--min-samples"));
    } else if (arg == "--max-mbytes-per-batch") {
      opt.max_bytes_per_batch =
        static_cast<std::size_t>(std::stoll(next("--max-mbytes-per-batch"))) * 1000000ull;
    } else if (arg == "--backend") {
      opt.backend = next("--backend");
    } else if (arg == "--index") {
      opt.index = next("--index");
    } else if (arg == "--neigh-per-row") {
      opt.neigh_per_row = std::stoi(next("--neigh-per-row"));
    } else if (arg == "--solo") {
      opt.solo = true;
    } else if (arg == "--perfil") {
      opt.perfil = true;
    } else if (arg == "--solo-isolado") {
      opt.solo         = true;
      opt.solo_isolado = true;
    } else if (arg == "--warmup") {
      opt.warmup = std::stoi(next("--warmup"));
    } else if (arg == "--repeat") {
      opt.repeat = std::stoi(next("--repeat"));
    } else if (arg == "--json") {
      opt.json = true;
    } else if (arg == "--selftest") {
      opt.selftest = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      std::exit(0);
    } else {
      std::cerr << "erro: argumento desconhecido '" << arg << "'\n";
      print_usage();
      return false;
    }
  }

  if (has_eps_list && opt.has_eps_range) {
    std::cerr << "erro: --eps e a faixa --eps-min/--eps-max/--eps-step são mutuamente "
                 "exclusivos\n";
    return false;
  }
  if (opt.repeat < 1) {
    std::cerr << "erro: --repeat deve ser >= 1\n";
    return false;
  }
  if (opt.warmup < 0) {
    std::cerr << "erro: --warmup deve ser >= 0\n";
    return false;
  }
  if (opt.has_eps_range) opt.eps = expand_eps_range(opt.eps_min, opt.eps_max, opt.eps_step);
  return true;
}

/** Ordena crescente e remove duplicatas — pré-condição dos Algoritmos 2 e 3. */
template <typename T>
void normalize_list(std::vector<T>& values, const char* flag)
{
  const std::vector<T> original = values;
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  if (values != original) {
    std::cerr << "aviso: " << flag
              << " foi ordenado/deduplicado; a ordem das configurações na saída segue os "
                 "campos \"eps\" e \"min_samples\" do JSON\n";
  }
}

std::vector<float> read_points(const std::string& path, int n, int d)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "erro: não foi possível abrir '" << path << "'\n";
    std::exit(1);
  }
  const std::streamsize bytes = f.tellg();
  const std::size_t expected  = static_cast<std::size_t>(n) * d * sizeof(float);
  if (static_cast<std::size_t>(bytes) != expected) {
    std::cerr << "erro: '" << path << "' tem " << bytes << " bytes, esperado " << expected
              << " para n=" << n << " d=" << d << " em float32\n";
    std::exit(1);
  }
  f.seekg(0);
  std::vector<float> data(static_cast<std::size_t>(n) * d);
  f.read(reinterpret_cast<char*>(data.data()), bytes);
  if (!f) {
    std::cerr << "erro: falha ao ler '" << path << "'\n";
    std::exit(1);
  }
  return data;
}

void write_labels(const std::string& path, const std::vector<int>& labels)
{
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "erro: não foi possível escrever '" << path << "'\n";
    std::exit(1);
  }
  f.write(reinterpret_cast<const char*>(labels.data()),
          static_cast<std::streamsize>(labels.size() * sizeof(int)));
  if (!f) {
    std::cerr << "erro: falha ao escrever '" << path << "'\n";
    std::exit(1);
  }
}

/** Mediana de uma amostra pequena (copia o vetor de propósito: a ordem original é reportada). */
float median_of(std::vector<float> values)
{
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  return (values.size() % 2) ? values[mid] : 0.5f * (values[mid - 1] + values[mid]);
}

/**
 * Executa o ajuste e devolve os rótulos no host, em ordem config-major.
 * O tempo medido cobre apenas a execução em GPU com os dados já residentes.
 *
 * `warmup` execuções são descartadas antes de medir. Isso importa porque a primeira
 * chamada do processo paga o carregamento do módulo CUDA e a materialização dos kernels,
 * custo que não é do algoritmo e que o baseline em Python não pagaria dentro da medição.
 * Em seguida `repeat` execuções são cronometradas individualmente; `timings` recebe todas,
 * na ordem em que ocorreram, e o chamador resume como preferir.
 */
template <typename Index_>
std::vector<int> fit_com_handle(const raft::handle_t& handle,
                                const std::vector<float>& host_points,
                                int n,
                                int d,
                                const std::vector<float>& eps,
                                const std::vector<int>& min_samples,
                                std::size_t max_bytes_per_batch,
                                ML::Dbscan::Multi::Backend backend,
                                int neigh_per_row,
                                int warmup,
                                int repeat,
                                std::vector<float>& timings,
                                rmm::device_uvector<char>* workspace = nullptr,
                                ML::Dbscan::Multi::PerfilMulti* perfil = nullptr)
{
  cudaStream_t stream = handle.get_stream();

  const int n_eps          = static_cast<int>(eps.size());
  const int n_min_pts      = static_cast<int>(min_samples.size());
  const std::size_t n_cfg  = static_cast<std::size_t>(n_eps) * n_min_pts;

  // O kernel compara distâncias ao quadrado; os raios são elevados aqui, uma única vez.
  std::vector<float> eps2(eps.size());
  std::transform(eps.begin(), eps.end(), eps2.begin(), [](float e) { return e * e; });

  const std::vector<Index_> min_pts_idx(min_samples.begin(), min_samples.end());

  rmm::device_uvector<float> d_points(host_points.size(), stream);
  raft::update_device(d_points.data(), host_points.data(), host_points.size(), stream);

  rmm::device_uvector<float> d_eps2(eps2.size(), stream);
  raft::update_device(d_eps2.data(), eps2.data(), eps2.size(), stream);

  rmm::device_uvector<Index_> d_min_pts(min_pts_idx.size(), stream);
  raft::update_device(d_min_pts.data(), min_pts_idx.data(), min_pts_idx.size(), stream);

  rmm::device_uvector<Index_> d_labels(static_cast<std::size_t>(n) * n_cfg, stream);

  handle.sync_stream(stream);

  auto run_once = [&]() {
    ML::Dbscan::Multi::fit_multi<float, Index_>(handle,
                                                d_points.data(),
                                                static_cast<Index_>(n),
                                                static_cast<Index_>(d),
                                                d_eps2.data(),
                                                n_eps,
                                                d_min_pts.data(),
                                                n_min_pts,
                                                d_labels.data(),
                                                max_bytes_per_batch,
                                                stream,
                                                backend,
                                                static_cast<Index_>(neigh_per_row),
                                                workspace,
                                                perfil);
  };

  for (int i = 0; i < warmup; ++i) {
    run_once();
    handle.sync_stream(stream);
  }

  cudaEvent_t start, stop;
  RAFT_CUDA_TRY(cudaEventCreate(&start));
  RAFT_CUDA_TRY(cudaEventCreate(&stop));

  timings.clear();
  timings.reserve(static_cast<std::size_t>(repeat));
  for (int i = 0; i < repeat; ++i) {
    float ms = 0.0f;
    RAFT_CUDA_TRY(cudaEventRecord(start, stream));
    run_once();
    RAFT_CUDA_TRY(cudaEventRecord(stop, stream));
    RAFT_CUDA_TRY(cudaEventSynchronize(stop));
    RAFT_CUDA_TRY(cudaEventElapsedTime(&ms, start, stop));
    timings.push_back(ms);
  }

  RAFT_CUDA_TRY(cudaEventDestroy(start));
  RAFT_CUDA_TRY(cudaEventDestroy(stop));

  // Os rótulos são estreitados para int32 na saída: o contrato de arquivo é .i32 e todo
  // rótulo é menor que N. A cópia extra fica fora da região cronometrada.
  std::vector<Index_> raw(static_cast<std::size_t>(n) * n_cfg);
  raft::update_host(raw.data(), d_labels.data(), raw.size(), stream);
  handle.sync_stream(stream);
  return std::vector<int>(raw.begin(), raw.end());
}

/**
 * Ver fit_com_handle; cria um raft::handle_t próprio, como faria um chamador avulso.
 *
 * O workspace é criado aqui e reaproveitado entre o warmup e os repeats. Sem isso,
 * `fit_multi` aloca e libera dezenas de GB DENTRO da região cronometrada a cada repetição —
 * a adjacência densa sozinha é N x batch_size bytes, 15 GB em N=256000 — e alocar 10 GB
 * custou 14 ms nos jobs 4864/4865.
 *
 * O ponto não é o tempo em si, é a assimetria: o laço de configurações isoladas
 * (medir_solo_impl) já compartilhava um workspace, então só a GRADE pagava. E `ganho_puro`
 * é grade dividido por isoladas — a métrica saía subestimada por construção, justamente
 * onde se quer medir o ganho.
 */
template <typename Index_>
std::vector<int> fit_impl(const std::vector<float>& host_points,
                          int n,
                          int d,
                          const std::vector<float>& eps,
                          const std::vector<int>& min_samples,
                          std::size_t max_bytes_per_batch,
                          ML::Dbscan::Multi::Backend backend,
                          int neigh_per_row,
                          int warmup,
                          int repeat,
                          std::vector<float>& timings,
                          bool reusar_workspace = true)
{
  raft::handle_t handle;
  rmm::device_uvector<char> workspace(0, handle.get_stream());
  return fit_com_handle<Index_>(handle,
                                host_points,
                                n,
                                d,
                                eps,
                                min_samples,
                                max_bytes_per_batch,
                                backend,
                                neigh_per_row,
                                warmup,
                                repeat,
                                timings,
                                reusar_workspace ? &workspace : nullptr);
}

/**
 * Uma execução instrumentada, para o --perfil. Separada de `fit` de propósito: warmup e
 * repeat não fazem sentido aqui (cada fase é cronometrada com sincronização, então repetir
 * só multiplica o custo) e o resultado nunca deve entrar em fit_ms.
 */
template <typename Index_>
void fit_perfilado_impl(const std::vector<float>& host_points,
                        int n,
                        int d,
                        const std::vector<float>& eps,
                        const std::vector<int>& min_samples,
                        std::size_t max_bytes_per_batch,
                        ML::Dbscan::Multi::Backend backend,
                        int neigh_per_row,
                        std::vector<float>& timings,
                        ML::Dbscan::Multi::PerfilMulti& perfil)
{
  raft::handle_t handle;
  fit_com_handle<Index_>(handle, host_points, n, d, eps, min_samples, max_bytes_per_batch,
                         backend, neigh_per_row, /* warmup */ 1, /* repeat */ 1, timings,
                         /* workspace */ nullptr, &perfil);
}

/** Tipo de índice do DBSCAN. `Auto` decide a partir de N — ver `int32_limita_lote`. */
enum class IndexType { Auto, Int32, Int64 };

/**
 * O tamanho de lote do cuML é limitado por `MAX_LABEL / N`, porque o CSR de um lote indexa
 * `N * batch_size` elementos. Com int32 isso trava o lote assim que `N² >= INT_MAX`, ou
 * seja, a partir de N ≈ 46341 — e cada lote a mais significa recalcular a vizinhança na
 * segunda passagem e uma rodada extra de fusão de rótulos POR CONFIGURAÇÃO. É o mesmo
 * aviso que o cuML imprime ("Using the larger integer type might result in better
 * performance").
 */
bool int32_limita_lote(int n)
{
  return static_cast<long long>(n) * n >= static_cast<long long>(std::numeric_limits<int>::max());
}

bool usa_int64(IndexType tipo, int n)
{
  return tipo == IndexType::Int64 || (tipo == IndexType::Auto && int32_limita_lote(n));
}

IndexType parse_index_type(const std::string& nome)
{
  if (nome == "auto") return IndexType::Auto;
  if (nome == "int32") return IndexType::Int32;
  if (nome == "int64") return IndexType::Int64;
  std::cerr << "erro: --index deve ser 'auto', 'int32' ou 'int64', recebi '" << nome << "'\n";
  std::exit(2);
}

/** Ver fit_perfilado_impl; despacha entre as instanciações de índice. */
void fit_perfilado(const std::vector<float>& host_points,
                   int n,
                   int d,
                   const std::vector<float>& eps,
                   const std::vector<int>& min_samples,
                   std::size_t max_bytes_per_batch,
                   ML::Dbscan::Multi::Backend backend,
                   IndexType index_type,
                   int neigh_per_row,
                   std::vector<float>& timings,
                   ML::Dbscan::Multi::PerfilMulti& perfil)
{
  if (usa_int64(index_type, n)) {
    fit_perfilado_impl<std::int64_t>(host_points, n, d, eps, min_samples, max_bytes_per_batch,
                                     backend, neigh_per_row, timings, perfil);
  } else {
    fit_perfilado_impl<int>(host_points, n, d, eps, min_samples, max_bytes_per_batch,
                            backend, neigh_per_row, timings, perfil);
  }
}

/** Ver fit_impl; despacha entre as instanciações de índice. */
std::vector<int> fit(const std::vector<float>& host_points,
                     int n,
                     int d,
                     const std::vector<float>& eps,
                     const std::vector<int>& min_samples,
                     std::size_t max_bytes_per_batch,
                     ML::Dbscan::Multi::Backend backend,
                     IndexType index_type,
                     int neigh_per_row,
                     int warmup,
                     int repeat,
                     std::vector<float>& timings)
{
  if (usa_int64(index_type, n)) {
    return fit_impl<std::int64_t>(host_points,
                                  n,
                                  d,
                                  eps,
                                  min_samples,
                                  max_bytes_per_batch,
                                  backend,
                                  neigh_per_row,
                                  warmup,
                                  repeat,
                                  timings);
  }
  return fit_impl<int>(host_points,
                       n,
                       d,
                       eps,
                       min_samples,
                       max_bytes_per_batch,
                       backend,
                       neigh_per_row,
                       warmup,
                       repeat,
                       timings);
}

/**
 * Mede cada configuração isolada (k=1, l=1), no mesmo processo.
 *
 * `handle_unico` decide o que se compartilha entre as configurações. É um experimento, não
 * uma preferência: o job 4862 mostrou que, com um raft::handle_t novo por chamada, a
 * segunda configuração em diante custa 2,3x a primeira — mesmo quando o trabalho é
 * idêntico (variar só minPts não muda a busca de vizinhança). Com handle compartilhado, se
 * a degradação sumir, a causa é a construção do handle; se persistir, é a alocação.
 */
template <typename Index_>
void medir_solo_impl(const std::vector<float>& points,
                     int n,
                     int d,
                     const std::vector<float>& eps,
                     const std::vector<int>& min_samples,
                     std::size_t max_bytes_per_batch,
                     ML::Dbscan::Multi::Backend backend,
                     int neigh_per_row,
                     int warmup,
                     int repeat,
                     bool handle_unico,
                     std::vector<std::vector<float>>& tempos_por_config)
{
  // O workspace é o que domina: a adjacência densa é N x batch_size bytes e chega a
  // dezenas de GB. Alocá-lo e liberá-lo por configuração foi a causa da degradação de
  // 2,3x medida no job 4863 — reaproveitá-lo é a correção, e é também o que qualquer
  // chamador em laço deveria fazer.
  raft::handle_t handle;
  rmm::device_uvector<char> workspace(0, handle.get_stream());

  for (std::size_t e = 0; e < eps.size(); ++e) {
    for (std::size_t m = 0; m < min_samples.size(); ++m) {
      const std::vector<float> eps1{eps[e]};
      const std::vector<int> mp1{min_samples[m]};
      std::vector<float> t;

      if (handle_unico) {
        // Handle e workspace compartilhados: é assim que se deve chamar em laço.
        fit_com_handle<Index_>(handle, points, n, d, eps1, mp1, max_bytes_per_batch,
                               backend, neigh_per_row, warmup, repeat, t, &workspace);
      } else {
        // Tudo novo a cada configuração, para reproduzir o comportamento antigo.
        fit_impl<Index_>(points, n, d, eps1, mp1, max_bytes_per_batch, backend,
                         neigh_per_row, warmup, repeat, t,
                         /* reusar_workspace */ false);
      }
      tempos_por_config.push_back(t);
    }
  }
}

void medir_solo(const std::vector<float>& points,
                int n,
                int d,
                const std::vector<float>& eps,
                const std::vector<int>& min_samples,
                std::size_t max_bytes_per_batch,
                ML::Dbscan::Multi::Backend backend,
                IndexType index_type,
                int neigh_per_row,
                int warmup,
                int repeat,
                bool handle_unico,
                std::vector<std::vector<float>>& tempos_por_config)
{
  if (usa_int64(index_type, n)) {
    medir_solo_impl<std::int64_t>(points, n, d, eps, min_samples, max_bytes_per_batch, backend,
                                  neigh_per_row, warmup, repeat, handle_unico,
                                  tempos_por_config);
  } else {
    medir_solo_impl<int>(points, n, d, eps, min_samples, max_bytes_per_batch, backend,
                         neigh_per_row, warmup, repeat, handle_unico, tempos_por_config);
  }
}

template <typename T>
std::string join(const std::vector<T>& values)
{
  std::ostringstream os;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) os << ",";
    os << values[i];
  }
  return os.str();
}

/** Número de clusters e de pontos de ruído de uma configuração. */
void summarize(const std::vector<int>& labels,
               int n,
               std::size_t config,
               int& n_clusters,
               int& n_noise)
{
  n_clusters   = 0;
  n_noise      = 0;
  const int* p = labels.data() + config * static_cast<std::size_t>(n);
  for (int i = 0; i < n; ++i) {
    if (p[i] < 0) {
      ++n_noise;
    } else if (p[i] + 1 > n_clusters) {
      n_clusters = p[i] + 1;
    }
  }
}

/**
 * Três blobs gaussianos bem separados nas duas primeiras dimensões; as demais ficam em
 * zero.
 *
 * O preenchimento com zeros é proposital: dimensões constantes não contribuem para a
 * distância ao quadrado, então a geometria — e portanto a resposta esperada — é idêntica
 * para qualquer D. Isso permite variar D só para exercitar os ramos especializados do
 * kernel (MAX_D = 4, 8, 16, 32 e o genérico) mantendo uma asserção exata: três clusters.
 */
std::vector<float> make_blobs(int n_per_cluster, int d, unsigned seed = 42)
{
  const int n               = 3 * n_per_cluster;
  const float centers[3][2] = {{0.0f, 0.0f}, {10.0f, 0.0f}, {0.0f, 10.0f}};

  std::mt19937 rng(seed);
  std::normal_distribution<float> noise(0.0f, 0.5f);
  std::vector<float> points(static_cast<std::size_t>(n) * d, 0.0f);
  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < n_per_cluster; ++i) {
      const std::size_t row = static_cast<std::size_t>(c) * n_per_cluster + i;
      points[row * d + 0]   = centers[c][0] + noise(rng);
      points[row * d + 1]   = centers[c][1] + noise(rng);
    }
  }
  return points;
}

/**
 * Renumera os clusters de uma configuração pela ordem de primeira aparição, preservando
 * -1. Rótulos de DBSCAN são invariantes a permutação: sem canonizar, duas partições iguais
 * poderiam ser declaradas diferentes só por terem numerado os grupos em outra ordem.
 */
std::vector<int> canonicalize(const std::vector<int>& labels, int n, std::size_t config)
{
  const int* p = labels.data() + config * static_cast<std::size_t>(n);
  std::vector<int> out(static_cast<std::size_t>(n), -1);
  std::vector<int> mapa(static_cast<std::size_t>(n), -1);
  int proximo = 0;
  for (int i = 0; i < n; ++i) {
    if (p[i] < 0) continue;
    if (mapa[static_cast<std::size_t>(p[i])] < 0) mapa[static_cast<std::size_t>(p[i])] = proximo++;
    out[static_cast<std::size_t>(i)] = mapa[static_cast<std::size_t>(p[i])];
  }
  return out;
}

/**
 * Verificação rápida sem dependência de Python, em três partes:
 *
 *   1. grade 3x3 sobre três blobs bem separados, checando as duas monotonicidades que a
 *      implementação multiparamétrica assume — com ε fixo o ruído não diminui quando
 *      minPts cresce; com minPts fixo o ruído não aumenta quando ε cresce;
 *   2. a mesma grade com o lote forçado a ser pequeno, exigindo rótulos idênticos aos do
 *      caso de lote único. É o que exercita a fusão de rótulos entre lotes, que em dados
 *      pequenos nunca roda e é a parte mais delicada herdada do cuML;
 *   3. varredura de D e de k, para percorrer os ramos especializados do kernel de
 *      vizinhança em vez de deixá-los quebrar na primeira execução com dados reais.
 */
int run_selftest(bool json, ML::Dbscan::Multi::Backend backend)
{
  const bool use_cuvs     = ML::Dbscan::Multi::backend_uses_cuvs(backend);
  const int n_per_cluster = 1000;
  const int n             = 3 * n_per_cluster;
  const int d             = 2;
  const std::vector<float> eps{0.3f, 0.5f, 0.8f};
  const std::vector<int> min_samples{4, 8, 16};

  const std::vector<float> points = make_blobs(n_per_cluster, d);

  std::vector<float> timings;
  const std::vector<int> labels =
    fit(points, n, d, eps, min_samples, 0, backend, IndexType::Int32, 0, 0, 1, timings);
  const float fit_ms = timings.front();

  const int k = static_cast<int>(eps.size());
  const int l = static_cast<int>(min_samples.size());

  std::vector<int> clusters(static_cast<std::size_t>(k) * l);
  std::vector<int> noise_counts(static_cast<std::size_t>(k) * l);
  std::ostringstream configs;

  for (int e = 0; e < k; ++e) {
    for (int m = 0; m < l; ++m) {
      const std::size_t config = static_cast<std::size_t>(e) * l + m;
      summarize(labels, n, config, clusters[config], noise_counts[config]);
      std::cerr << "[selftest] eps=" << eps[e] << " minPts=" << min_samples[m]
                << " clusters=" << clusters[config] << " ruido=" << noise_counts[config] << "\n";
      if (config) configs << ",";
      configs << "{\"eps\":" << eps[e] << ",\"min_samples\":" << min_samples[m]
              << ",\"n_clusters\":" << clusters[config] << ",\"n_noise\":" << noise_counts[config]
              << "}";
    }
  }

  bool ok = true;

  // O maior ε com o menor minPts deve enxergar os três blobs.
  const std::size_t easiest = static_cast<std::size_t>(k - 1) * l + 0;
  if (clusters[easiest] != 3) {
    std::cerr << "[selftest] FALHA: esperava 3 clusters em eps=" << eps[k - 1]
              << " minPts=" << min_samples[0] << ", obtive " << clusters[easiest] << "\n";
    ok = false;
  }

  for (int e = 0; e < k; ++e) {
    for (int m = 1; m < l; ++m) {
      const std::size_t cur  = static_cast<std::size_t>(e) * l + m;
      const std::size_t prev = static_cast<std::size_t>(e) * l + (m - 1);
      if (noise_counts[cur] < noise_counts[prev]) {
        std::cerr << "[selftest] FALHA: com eps=" << eps[e] << " o ruído caiu de "
                  << noise_counts[prev] << " para " << noise_counts[cur]
                  << " ao aumentar minPts\n";
        ok = false;
      }
    }
  }

  for (int m = 0; m < l; ++m) {
    for (int e = 1; e < k; ++e) {
      const std::size_t cur  = static_cast<std::size_t>(e) * l + m;
      const std::size_t prev = static_cast<std::size_t>(e - 1) * l + m;
      if (noise_counts[cur] > noise_counts[prev]) {
        std::cerr << "[selftest] FALHA: com minPts=" << min_samples[m] << " o ruído subiu de "
                  << noise_counts[prev] << " para " << noise_counts[cur] << " ao aumentar eps\n";
        ok = false;
      }
    }
  }

  // --- parte 2: múltiplos lotes ---------------------------------------------
  // Em dados pequenos o lote cobre N inteiro e a fusão de rótulos entre lotes nunca roda,
  // apesar de ser a parte mais delicada herdada do cuML. Forçando um orçamento apertado, o
  // resultado tem de ser o MESMO: o particionamento em lotes é detalhe de execução, não do
  // algoritmo. Uma divergência aqui aponta para a fusão, não para a vizinhança.
  const std::size_t alvo_lote = static_cast<std::size_t>(n) / 4;
  const std::size_t orcamento =
    ML::Dbscan::Multi::max_bytes_for_batch_size<int>(n, k, l, alvo_lote, use_cuvs);

  std::size_t memoria_estimada = 0;
  const std::size_t lote_efetivo = ML::Dbscan::Multi::compute_batch_size_multi<int>(
    memoria_estimada, n, k, l, orcamento, use_cuvs);
  const int n_lotes = static_cast<int>((n + lote_efetivo - 1) / lote_efetivo);

  std::vector<float> timings_lotes;
  const std::vector<int> labels_lotes =
    fit(points, n, d, eps, min_samples, orcamento, backend, IndexType::Int32, 0, 0, 1,
        timings_lotes);

  int configs_divergentes = 0;
  for (std::size_t config = 0; config < static_cast<std::size_t>(k) * l; ++config) {
    if (canonicalize(labels, n, config) != canonicalize(labels_lotes, n, config)) {
      ++configs_divergentes;
      int c_ref = 0, r_ref = 0, c_lot = 0, r_lot = 0;
      summarize(labels, n, config, c_ref, r_ref);
      summarize(labels_lotes, n, config, c_lot, r_lot);
      std::cerr << "[selftest] FALHA: config " << config << " diverge entre 1 lote e "
                << n_lotes << " lotes (clusters " << c_ref << " vs " << c_lot << ", ruído "
                << r_ref << " vs " << r_lot << ")\n";
      ok = false;
    }
  }
  std::cerr << "[selftest] lotes: " << n_lotes << " x " << lote_efetivo << " pontos, "
            << (static_cast<std::size_t>(k) * l - configs_divergentes) << "/"
            << (static_cast<std::size_t>(k) * l) << " configurações idênticas ao lote único\n";
  if (n_lotes < 2) {
    std::cerr << "[selftest] FALHA: o orçamento apertado não produziu mais de um lote; a "
                 "fusão de rótulos não foi exercitada\n";
    ok = false;
  }

  // --- parte 3: ramos especializados do kernel -------------------------------
  // D escolhe entre os kernels de registrador (MAX_D = 4, 8, 16, 32) e o genérico; k
  // escolhe entre MAX_K = 1, 4 e 16. Sem esta varredura, um ramo com erro só apareceria na
  // primeira execução com dados reais daquela dimensionalidade.
  const int dims_teste[]       = {2, 8, 16, 32, 33};
  const std::vector<std::vector<float>> eps_teste{
    {0.8f}, {0.3f, 0.5f, 0.8f}, {0.2f, 0.3f, 0.5f, 0.65f, 0.8f}};
  const std::vector<int> min_pts_teste{4};
  const int n_pc_teste = 500;
  const int n_teste    = 3 * n_pc_teste;

  for (int dt : dims_teste) {
    const std::vector<float> pts = make_blobs(n_pc_teste, dt);
    bool dim_ok                  = true;
    for (const auto& eps_t : eps_teste) {
      std::vector<float> t;
      const std::vector<int> lab =
        fit(pts, n_teste, dt, eps_t, min_pts_teste, 0, backend, IndexType::Int32, 0, 0, 1, t);

      // O maior ε é o último; com minPts=4 tem de enxergar exatamente os três blobs.
      const std::size_t maior = eps_t.size() - 1;
      int c = 0, r = 0;
      summarize(lab, n_teste, maior, c, r);
      if (c != 3) {
        std::cerr << "[selftest] FALHA: D=" << dt << " k=" << eps_t.size()
                  << " esperava 3 clusters em eps=" << eps_t.back() << ", obtive " << c << "\n";
        ok = dim_ok = false;
      }

      // Monotonicidade em ε dentro deste k, que é o que valida os códigos daquele ramo.
      int r_anterior = -1;
      for (std::size_t e = 0; e < eps_t.size(); ++e) {
        int ce = 0, re = 0;
        summarize(lab, n_teste, e, ce, re);
        if (r_anterior >= 0 && re > r_anterior) {
          std::cerr << "[selftest] FALHA: D=" << dt << " k=" << eps_t.size()
                    << " ruído subiu de " << r_anterior << " para " << re << " ao aumentar eps\n";
          ok = dim_ok = false;
        }
        r_anterior = re;
      }
    }
    std::cerr << "[selftest] D=" << dt << (dim_ok ? " ok" : " FALHOU") << " para k = 1, 3 e 5\n";
  }

  // --- parte 4: tipo de índice ----------------------------------------------
  // int64 existe para destravar o tamanho do lote (int32 o limita a MAX_INT/N), e com
  // N grande é o caminho padrão. O tipo do índice é detalhe de representação: os rótulos
  // têm de ser os mesmos. Aqui N é pequeno, então esta é a única oportunidade de exercitar
  // a instanciação int64 com uma resposta conhecida.
  std::vector<float> timings_i64;
  const std::vector<int> labels_i64 =
    fit(points, n, d, eps, min_samples, 0, backend, IndexType::Int64, 0, 0, 1, timings_i64);

  int divergentes_i64 = 0;
  for (std::size_t config = 0; config < static_cast<std::size_t>(k) * l; ++config) {
    if (canonicalize(labels, n, config) != canonicalize(labels_i64, n, config)) {
      ++divergentes_i64;
      std::cerr << "[selftest] FALHA: config " << config << " diverge entre int32 e int64\n";
      ok = false;
    }
  }
  std::cerr << "[selftest] índice: " << (static_cast<std::size_t>(k) * l - divergentes_i64) << "/"
            << (static_cast<std::size_t>(k) * l) << " configurações idênticas entre int32 e int64"
            << " (auto para N=" << n << " escolheria "
            << (usa_int64(IndexType::Auto, n) ? "int64" : "int32") << ")\n";

  // --- parte 5: correção do lote quando neigh_per_row mente -------------------
  // `--neigh-per-row` dimensiona o lote ANTES da busca, então é uma promessa sobre os
  // dados. Os jobs 4866-4871 morreram em `std::bad_alloc` porque a promessa (512) estava
  // 110x abaixo do grau real de heterogeneous_blobs: o lote saiu do tamanho do dataset e o
  // CSR não coube. Agora o runner mede o grau, aborta antes de alocar e refaz o lote.
  //
  // Aqui o teto artificial faz o papel da GPU cheia: um palpite de 1 vizinho por linha
  // produz o lote máximo, e o teto obriga a correção a acontecer. O que se exige é que os
  // rótulos saiam idênticos aos da execução sem aperto — a correção muda o lote, não a
  // resposta.
  //
  // 1 MB: apertado o bastante para o lote de 3000 linhas estourar em qualquer densidade
  // plausível, e folgado o bastante para o lote corrigido ficar na casa das dezenas de
  // linhas. Um teto muito menor levaria o pior caso a lotes de 1 linha e o teste passaria
  // a gastar minutos em 3000 fusões de rótulos.
  const std::size_t teto_anterior = ML::Dbscan::Multi::csr_teto_de_teste();
  ML::Dbscan::Multi::csr_correcoes_de_lote() = 0;
  ML::Dbscan::Multi::csr_teto_de_teste()     = 1024u * 1024u;

  std::vector<float> timings_ajuste;
  const std::vector<int> labels_ajuste =
    fit(points, n, d, eps, min_samples, 0, backend, IndexType::Int32,
        /* neigh_per_row */ 1, 0, 1, timings_ajuste);

  ML::Dbscan::Multi::csr_teto_de_teste() = teto_anterior;

  const int correcoes = ML::Dbscan::Multi::csr_correcoes_de_lote();
  if (correcoes == 0) {
    std::cerr << "[selftest] FALHA: o teto apertado não obrigou nenhuma correção de lote; "
                 "o teste não exercitou nada\n";
    ok = false;
  }

  int divergentes_ajuste = 0;
  for (std::size_t config = 0; config < static_cast<std::size_t>(k) * l; ++config) {
    if (canonicalize(labels, n, config) != canonicalize(labels_ajuste, n, config)) {
      ++divergentes_ajuste;
      std::cerr << "[selftest] FALHA: config " << config
                << " diverge depois da correção do lote\n";
      ok = false;
    }
  }
  std::cerr << "[selftest] lote corrigido (" << correcoes << "x): "
            << (static_cast<std::size_t>(k) * l - divergentes_ajuste) << "/"
            << (static_cast<std::size_t>(k) * l)
            << " configurações idênticas com neigh-per-row mentindo (1) e o CSR apertado\n";

  // --- parte 6: as duas rotas do multi-eps ----------------------------------
  // A partir do job 4895, a rota é escolhida pelo nnz medido: grafo esparso anota o CSR do
  // maior raio e compacta; grafo denso reconstrói um CSR por raio chamando o cuVS de novo.
  // A escolha é de DESEMPENHO — anotar custava 68% do tempo em dados densos, e era o que
  // fazia o multi perder para o cuML. Os rótulos têm de ser os mesmos nas duas.
  //
  // Sem forçar, o dataset do teste escolheria uma só e a outra nunca rodaria.
  std::vector<int> labels_rota[2];
  for (int rota = 1; rota <= 2; ++rota) {
    ML::Dbscan::Multi::rota_forcada() = rota;
    std::vector<float> t_rota;
    labels_rota[rota - 1] = fit(points, n, d, eps, min_samples, 0, backend, IndexType::Int32,
                                0, 0, 1, t_rota);
  }
  ML::Dbscan::Multi::rota_forcada() = 0;

  int divergentes_rota = 0;
  for (std::size_t config = 0; config < static_cast<std::size_t>(k) * l; ++config) {
    if (canonicalize(labels_rota[0], n, config) != canonicalize(labels_rota[1], n, config)) {
      ++divergentes_rota;
      std::cerr << "[selftest] FALHA: config " << config
                << " diverge entre a rota que anota e a rota densa\n";
      ok = false;
    }
  }
  std::cerr << "[selftest] rotas: " << (static_cast<std::size_t>(k) * l - divergentes_rota) << "/"
            << (static_cast<std::size_t>(k) * l)
            << " configuracoes identicas entre anotar-e-compactar e um CSR por raio\n";

  std::cerr << "[selftest] " << (ok ? "PASSOU" : "FALHOU") << "\n";

  if (json) {
    std::cout << "{\"selftest\":" << (ok ? "true" : "false")
              << ",\"backend\":\"" << (use_cuvs ? "cuvs" : "codes") << "\""
              << ",\"fit_ms\":" << fit_ms
              << ",\"n\":" << n << ",\"d\":" << d << ",\"eps\":[" << join(eps)
              << "],\"min_samples\":[" << join(min_samples)
              << "],\"configuration_count\":" << (k * l)
              << ",\"config_order\":\"eps_major\",\"batches\":" << n_lotes
              << ",\"batch_size\":" << lote_efetivo
              << ",\"batched_mismatches\":" << configs_divergentes
              << ",\"index_mismatches\":" << divergentes_i64
              << ",\"batch_fixup_mismatches\":" << divergentes_ajuste
              << ",\"route_mismatches\":" << divergentes_rota
              << ",\"configurations\":[" << configs.str() << "]}" << std::endl;
  }
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv)
{
  Options opt;
  if (!parse_args(argc, argv, opt)) return 2;

  const ML::Dbscan::Multi::Backend backend = parse_backend(opt.backend);
  const IndexType index_type               = parse_index_type(opt.index);

  if (opt.selftest) return run_selftest(opt.json, backend);

  if (opt.input.empty() || opt.n <= 0 || opt.d <= 0 || opt.eps.empty() ||
      opt.min_samples.empty()) {
    std::cerr << "erro: --input, --n, --d, --eps (ou a faixa) e --min-samples são "
                 "obrigatórios\n";
    print_usage();
    return 2;
  }
  normalize_list(opt.eps, "--eps");
  normalize_list(opt.min_samples, "--min-samples");

  if (opt.eps.front() <= 0.0f) {
    std::cerr << "erro: valores de eps devem ser positivos\n";
    return 2;
  }
  if (opt.eps.size() > static_cast<std::size_t>(ML::Dbscan::Multi::VertexDeg::kMaxEps)) {
    std::cerr << "erro: no máximo " << ML::Dbscan::Multi::VertexDeg::kMaxEps
              << " valores de eps por execução\n";
    return 2;
  }

  const std::vector<float> points = read_points(opt.input, opt.n, opt.d);

  std::vector<float> timings;
  const std::vector<int> labels = fit(points,
                                      opt.n,
                                      opt.d,
                                      opt.eps,
                                      opt.min_samples,
                                      opt.max_bytes_per_batch,
                                      backend,
                                      index_type,
                                      opt.neigh_per_row,
                                      opt.warmup,
                                      opt.repeat,
                                      timings);
  const float fit_ms            = median_of(timings);

  if (!opt.output.empty()) write_labels(opt.output, labels);

  // O perfil roda UMA passagem extra, depois da medição e fora dela: os eventos por fase
  // sincronizam o stream e contaminariam fit_ms. Serve para localizar o gargalo, não para
  // reportar tempo.
  if (opt.perfil) {
    ML::Dbscan::Multi::PerfilMulti perf;
    std::vector<float> descartado;
    fit_perfilado(points, opt.n, opt.d, opt.eps, opt.min_samples, opt.max_bytes_per_batch,
                  backend, index_type, opt.neigh_per_row, descartado, perf);

    const std::size_t n_cfg = opt.eps.size() * opt.min_samples.size();
    const double tot = perf.total() > 0 ? perf.total() : 1.0;
    // `por_config` é o que decide o teto do ganho, então é campo explícito e não texto:
    // distinguir por substring dependia de acento e de caixa, e "relabel final" caía no
    // lado errado justamente por isso.
    struct Fase {
      const char* nome;
      double ms;
      const char* quando;
      bool por_config;
    };
    const Fase fases[] = {
      {"busca (passagem 1)", perf.busca_p1, "por lote", false},
      {"CSR (passagem 1)", perf.csr_p1, "por lote", false},
      {"anotacao (passagem 1)", perf.anotacao_p1, "por lote", false},
      {"pontos centrais", perf.nucleo, "por lote", false},
      {"busca (passagem 2)", perf.busca_p2, "por lote > 0", false},
      {"CSR (passagem 2)", perf.csr_p2, "por lote", false},
      {"anotacao (passagem 2)", perf.anotacao_p2, "por lote", false},
      {"filtro por eps", perf.filtro, "por lote e eps", false},
      {"rotulagem (weak_cc)", perf.rotulagem, "POR CONFIGURACAO", true},
      {"fusao de rotulos", perf.fusao, "POR CONFIGURACAO", true},
      {"relabel final", perf.relabel, "POR CONFIGURACAO", true},
    };

    std::fprintf(stderr,
                 "\n[perfil] %d lote(s), nnz maximo %lld, %zu configuracoes\n",
                 perf.n_lotes,
                 perf.nnz_max,
                 n_cfg);
    std::fprintf(stderr, "[perfil] %-22s %8s %6s   %s\n", "fase", "ms", "%", "quando");

    double compartilhado = 0, por_config = 0;
    for (const Fase& f : fases) {
      std::fprintf(
        stderr, "[perfil] %-22s %8.1f %5.1f%%   %s\n", f.nome, f.ms, 100.0 * f.ms / tot, f.quando);
      if (f.por_config) {
        por_config += f.ms;
      } else {
        compartilhado += f.ms;
      }
    }

    std::fprintf(stderr, "[perfil] %-22s %8.1f\n", "total", perf.total());

    // Os tempos acima somam TODAS as passagens de fit_multi que couberam no perfil, warmup
    // inclusive. Ler um deles como o custo de uma operação única foi exatamente o erro que
    // me fez enxergar uma anomalia no weak_cc que não existia: `busca` cobre 2 execuções e
    // k raios, não uma busca. A abertura abaixo já sai normalizada.
    const int execs = perf.execucoes > 0 ? perf.execucoes : 1;
    std::fprintf(stderr,
                 "\n[perfil] rotulagem por raio, POR EXECUCAO (%d execucao(oes) no total)\n",
                 execs);
    std::fprintf(stderr,
                 "[perfil] %5s %12s %14s %10s %12s\n",
                 "eps#",
                 "chamadas",
                 "arestas",
                 "ms",
                 "ms/G-aresta");
    for (int e = 0; e < static_cast<int>(opt.eps.size())
                    && e < ML::Dbscan::Multi::PerfilMulti::kMaxEpsPerfil;
         ++e) {
      const double ms       = perf.rotulagem_eps[e] / execs;
      const double arestas  = static_cast<double>(perf.nnz_rotulado[e]) / execs;
      const double por_giga = arestas > 0 ? ms / (arestas / 1e9) : 0.0;
      std::fprintf(stderr,
                   "[perfil] %5d %12.1f %14.3e %10.1f %12.1f\n",
                   e,
                   static_cast<double>(perf.chamadas_eps[e]) / execs,
                   arestas,
                   ms,
                   por_giga);
    }
    // Na rota densa a passagem 1 chama o cuVS uma vez por raio; na que anota, uma só. Sem
    // saber a rota, dividir busca_p1 pelo número de raios daria o valor errado por 3x.
    const int buscas_p1_por_exec = perf.lotes_densos > 0 ? static_cast<int>(opt.eps.size()) : 1;
    std::fprintf(stderr,
                 "[perfil] rota: %d lote(s) densos, %d anotado(s) (por execucao)\n",
                 perf.lotes_densos / execs,
                 perf.lotes_anotados / execs);
    std::fprintf(stderr,
                 "[perfil] uma varredura N^2*D custa ~%.1f ms\n",
                 perf.busca_p1 / execs / buscas_p1_por_exec);
    std::fprintf(stderr,
                 "[perfil] compartilhado entre configuracoes: %8.1f ms (%.1f%%)\n"
                 "[perfil] pago por configuracao:             %8.1f ms (%.1f ms cada)\n"
                 "[perfil] teto do ganho = total/por_config = %.1fx\n",
                 compartilhado,
                 100.0 * compartilhado / tot,
                 por_config,
                 por_config / static_cast<double>(n_cfg),
                 por_config > 0 ? perf.total() / por_config : 0.0);
  }

  // Cada configuração isolada, no MESMO processo. Compartilhar o processo é o que torna
  // esta medição comparável com a do baseline, que percorre as configurações num processo
  // Python só e portanto já roda com contexto CUDA quente e clocks em boost. Tudo o mais
  // continua isolado: handle, upload, workspace, warmup e repeat são refeitos por
  // configuração.
  std::vector<float> solo_ms;
  std::ostringstream solo_all;
  if (opt.solo) {
    std::vector<std::vector<float>> tempos;
    medir_solo(points,
               opt.n,
               opt.d,
               opt.eps,
               opt.min_samples,
               opt.max_bytes_per_batch,
               backend,
               index_type,
               opt.neigh_per_row,
               opt.warmup,
               opt.repeat,
               !opt.solo_isolado,
               tempos);
    for (const auto& t : tempos) {
      if (!solo_ms.empty()) solo_all << ",";
      solo_all << "[" << join(t) << "]";
      solo_ms.push_back(median_of(t));
    }
  }

  if (opt.json) {
    const std::size_t n_cfg = opt.eps.size() * opt.min_samples.size();
    std::cout << "{\"fit_ms\":" << fit_ms << ",\"fit_ms_all\":[" << join(timings)
              << "],\"backend\":\"" << (ML::Dbscan::Multi::backend_uses_cuvs(backend) ? "cuvs"
                                                                                      : "codes")
              << "\",\"index\":\"" << (usa_int64(index_type, opt.n) ? "int64" : "int32")
              << "\",\"neigh_per_row\":" << opt.neigh_per_row
              << ",\"warmup\":" << opt.warmup << ",\"repeat\":" << opt.repeat
              << ",\"configuration_count\":" << n_cfg << ",\"eps_count\":" << opt.eps.size()
              << ",\"min_samples_count\":" << opt.min_samples.size() << ",\"eps\":["
              << join(opt.eps) << "],\"min_samples\":[" << join(opt.min_samples)
              << "],\"config_order\":\"eps_major\",\"n\":" << opt.n << ",\"d\":" << opt.d;
    if (opt.solo) {
      std::cout << ",\"solo_ms\":[" << join(solo_ms) << "],\"solo_ms_all\":[" << solo_all.str()
                << "]";
    }
    std::cout << "}" << std::endl;
  }
  return 0;
}
