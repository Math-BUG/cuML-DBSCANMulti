# Código vendorizado do cuML — registro de proveniência

Esta árvore contém um subconjunto **verbatim, não modificado** do código-fonte do
DBSCAN do cuML, copiado com os caminhos originais preservados. É a base a partir da
qual as variantes Multi-EPS, Multi-minPts e Multi-Both são derivadas.

**Regra:** nada aqui é editado. Modificações vivem fora desta pasta, para que
`diff -r` contra o upstream continue mostrando exatamente o que mudou.

## Origem

| Campo | Valor |
|-------|-------|
| Repositório | https://github.com/rapidsai/cuml |
| Tag | `v26.02.00` |
| Commit | `22b12c8c3e378f17f35107f7fb4ffd65a3dce534` (2026-02-05) |
| Objeto da tag | `955aa626ec1fcc6520f9708f8226fa9e032f9b8e` (tag anotada) |
| Licença | Apache-2.0 — ver [`LICENSE`](LICENSE) |
| Copyright | NVIDIA CORPORATION, 2018–2026 |
| Data da cópia | 2026-08-04 |

## Arquivos e verificação

Cada arquivo foi conferido contra o *blob SHA* do próprio git do upstream. Para
reverificar a qualquer momento, prefira o manifesto legível por máquina `VENDORED.json`
e `python scripts/check_repo_metadata.py`. A receita manual equivalente é:

```bash
gh api "repos/rapidsai/cuml/git/trees/v26.02.00?recursive=1" \
  --jq '.tree[] | select(.type=="blob") | .path + " " + .sha' > /tmp/blobs.txt
cd third_party/cuml
find . -type f -not -name VENDORED.md -not -name VENDORED.json | \
  sed 's|^\./||' | sort | while read f; do
  up=$(grep -E "^$f " /tmp/blobs.txt | awk '{print $2}')
  [ "$up" = "$(git hash-object "$f")" ] && echo "OK   $f" || echo "DIFF $f"
done
```

| blob SHA | Arquivo | Papel |
|----------|---------|-------|
| `3ba63d53f4bd` | `LICENSE` | Licença Apache-2.0 do cuML |
| `6c0e7ccafbb4` | `cpp/include/cuml/cluster/dbscan.hpp` | API pública, `EpsNnMethod` |
| `beda1f05e8bd` | `cpp/include/cuml/common/distance_type.hpp` | Enum de métricas |
| `cb902255f9cc` | `cpp/include/cuml/common/logger.hpp` | Logger |
| `475780bd015a` | `cpp/include/cuml/common/utils.hpp` | Macro `CUML_KERNEL` |
| `7addc81eae6c` | `cpp/src/common/nvtx.hpp` | Ranges NVTX |
| `00ce78f8e9fc` | `cpp/src/dbscan/dbscan.cu` | Instanciação de `fit` |
| `5911a09999c4` | `cpp/src/dbscan/dbscan.cuh` | `dbscanFitImpl`, `compute_batch_size` |
| `8e2e9b83c16f` | `cpp/src/dbscan/runner.cuh` | Orquestração das 5 etapas |
| `cf6a48d25564` | `cpp/src/dbscan/adjgraph/algo.cuh` | `exclusive_scan` + `adj_to_csr` |
| `8d8465c081e2` | `cpp/src/dbscan/adjgraph/pack.h` | Parâmetros da etapa |
| `98dd99b39fb4` | `cpp/src/dbscan/adjgraph/runner.cuh` | Dispatch da etapa |
| `17a2e2be1ce0` | `cpp/src/dbscan/corepoints/compute.cuh` | `vd >= min_pts` |
| `8134efe69560` | `cpp/src/dbscan/corepoints/exchange.cuh` | Troca multi-nó (fora de escopo) |
| `9811f7275d40` | `cpp/src/dbscan/mergelabels/runner.cuh` | Fusão de rótulos entre lotes |
| `80e1794ef90a` | `cpp/src/dbscan/mergelabels/tree_reduction.cuh` | Redução multi-nó (fora de escopo) |
| `649c20091b48` | `cpp/src/dbscan/vertexdeg/algo.cuh` | Vizinhança + graus (**a substituir**) |
| `35cf4d73edcf` | `cpp/src/dbscan/vertexdeg/pack.h` | Parâmetros da etapa |
| `bac81ab31065` | `cpp/src/dbscan/vertexdeg/precomputed.cuh` | Caminho `Precomputed` (fora de escopo) |
| `cd3a352d7737` | `cpp/src/dbscan/vertexdeg/runner.cuh` | Dispatch da etapa |
| `4cbe8c982b24` | `cpp/tests/sg/dbscan_test.cu` | Referência para os testes de equivalência |
| `cf63ae845bab` | `cpp/bench/sg/dbscan.cu` | Referência de medição |

Todos os 22 arquivos upstream conferiram **OK** na cópia de 2026-08-04. `VENDORED.md` e
`VENDORED.json` são metadados locais e não entram nessa contagem.

## Dependências externas destes arquivos

Levantadas a partir dos `#include` da árvore vendorizada:

| Dependência | Natureza | Situação |
|-------------|----------|----------|
| RAFT (`raft/core`, `raft/sparse`, `raft/linalg`, `raft/label`, `raft/util`) | Cabeçalhos; o projeto pode ligar `libraft` conforme o ambiente | Necessária |
| RMM (`rmm/device_uvector.hpp`) | Cabeçalhos + símbolos em `librmm.so` no build derivado | Necessária |
| Thrust / CUB / CCCL | CUDA Toolkit ou cópia compatível no wheel RMM | Necessária |
| **cuVS** (`cuvs/neighbors/epsilon_neighborhood.hpp`, `cuvs/neighbors/ball_cover.hpp`) | Biblioteca **compilada** | Usada em **apenas 2 arquivos**: `runner.cuh` e `vertexdeg/algo.cuh` |
| `rapids_logger/logger.hpp` | Headers + `librapids_logger.so` | Necessária transitivamente por `raft/core/logger.hpp`; o shim substitui apenas o logger do cuML |

**Consequência no projeto derivado atual.** A árvore acima permanece apenas como fonte
verbatim; o build usa os arquivos modificados em `src/multi/`. O backend padrão preserva a
chamada `epsilon_neighborhood::compute` e, portanto, compila e liga `libcuvs`. O backend
`codes` substitui a busca e não precisa de `libcuvs`. O shim em
`src/compat/cuml/common/logger.hpp` elimina o logger próprio do cuML, mas os headers do
RAFT ainda exigem `rapids_logger/log_levels.h`; por isso `rapids-logger` permanece. Em
nenhum dos casos é necessário construir o `libcuml++` completo.

No backend cuVS, a execução começa no maior ε e escolhe por lote entre anotar/compactar o
CSR (grafo esparso) e reutilizar esse primeiro resultado enquanto calcula os demais raios,
totalizando uma chamada cuVS por ε (grafo denso). Essa política pertence ao código derivado
e não altera os blobs registrados neste manifesto.

## Cabeçalho gerado em tempo de build

`cpp/include/cuml/common/logger.hpp` inclui `cuml/common/logger_macros.hpp`, que **não
existe no repositório**: é gerado pelo CMake do cuML via
`create_logger_macros(CUML "ML::default_logger()" include/cuml/common)`
(`cpp/CMakeLists.txt:259`). Em um build fora do CMake do cuML é preciso fornecer um
*shim* com as macros `CUML_LOG_DEBUG` / `CUML_LOG_INFO` / `CUML_LOG_WARN`. O projeto
fornece esse shim em `src/compat/cuml/common/logger.hpp`, fora desta pasta, como exige a
regra de árvore verbatim. Isso não elimina a dependência transitiva introduzida pelo RAFT.
As dependências
efetivamente linkadas continuam sendo definidas pelo `Makefile`; este manifesto não deve
ser usado como lista de link.

## Como atualizar a versão fixada

Trocar a tag invalida a comparação com o *baseline*: o código derivado e o pacote
`cuml` do baseline precisam ser da mesma versão. Ao atualizar:

1. rebaixar a tag em todos os lugares (aqui, no `docs/fontes-primarias.md` e no
   ambiente Python do baseline);
2. recopiar os arquivos e reverificar os *blob SHAs*;
3. refazer o `diff` das modificações — o upstream pode ter mexido no `runner.cuh`;
4. reexecutar o oráculo semântico de DBSCAN e a suíte de validação completa antes de
   qualquer medição de tempo.
