# cuML-DBSCANMulti — build do executável CUDA
#
# O binário é compilado direto com nvcc, sem o CMake do cuML e sem libcuml++/libcuvs.
# Bastam os wheels libraft/librmm/rapids_logger e o CUDA Toolkit: deles saem tanto os
# cabeçalhos quanto as .so do link (RMM não é header-only — rmm::cuda_stream_view,
# rmm::device_buffer e rmm::bad_alloc vivem em librmm.so).
#
# Ordem dos -I (importante): src/compat vem primeiro, para que o shim de
# <cuml/common/logger.hpp> seja encontrado antes do original vendorizado, que depende de
# um cabeçalho gerado pelo CMake do cuML e da biblioteca rapids_logger.
#
# Uso típico:
#   source .venv/bin/activate      # os cabeçalhos vêm dos wheels libraft/librmm
#   make check-headers             # diagnóstico de onde estão RAFT/RMM
#   make CUDA_ARCH=sm_80           # A100
#   make selftest                  # roda a verificação embutida (precisa de GPU)

NVCC      ?= nvcc
PYTHON    ?= python3
CUDA_ARCH ?= native
STD       ?= c++17
BUILD_DIR ?= build
TARGET    ?= $(BUILD_DIR)/dbscan_multi

# Cabeçalhos de RAFT/RMM, procurados em três lugares, nesta ordem:
#  1. RAPIDS_INCLUDE=<prefixo>/include, passado à mão (ambiente conda);
#  2. third_party/rapids_headers/, criado por scripts/fetch_rapids_headers.sh
#     (não exige pip nem Python >= 3.11 — é só extrair o zip do wheel);
#  3. wheels libraft-cu12/librmm-cu12 instalados no venv ativo.
RAPIDS_INCLUDE ?=
VENDORED_HEADERS := third_party/rapids_headers

ifneq ($(strip $(RAPIDS_INCLUDE)),)
  RAFT_INCLUDE   ?= $(RAPIDS_INCLUDE)
  RMM_INCLUDE    ?= $(RAPIDS_INCLUDE)
  LOGGER_INCLUDE ?= $(RAPIDS_INCLUDE)
  CUVS_INCLUDE   ?= $(RAPIDS_INCLUDE)
  RMM_LIBDIR     ?= $(firstword $(wildcard $(RAPIDS_INCLUDE)/../lib64 $(RAPIDS_INCLUDE)/../lib))
  RAFT_LIBDIR    ?= $(RMM_LIBDIR)
  LOGGER_LIBDIR  ?= $(RMM_LIBDIR)
  CUVS_LIBDIR    ?= $(RMM_LIBDIR)
else ifneq ($(wildcard $(VENDORED_HEADERS)/libraft/include/raft/core/handle.hpp),)
  RAFT_INCLUDE   ?= $(VENDORED_HEADERS)/libraft/include
  RMM_INCLUDE    ?= $(VENDORED_HEADERS)/librmm/include
  LOGGER_INCLUDE ?= $(VENDORED_HEADERS)/rapids_logger/include
  CUVS_INCLUDE   ?= $(firstword $(wildcard $(VENDORED_HEADERS)/libcuvs/include))
  RMM_LIBDIR     ?= $(firstword $(wildcard $(VENDORED_HEADERS)/librmm/lib64))
  RAFT_LIBDIR    ?= $(firstword $(wildcard $(VENDORED_HEADERS)/libraft/lib64))
  LOGGER_LIBDIR  ?= $(firstword $(wildcard $(VENDORED_HEADERS)/rapids_logger/lib64))
  CUVS_LIBDIR    ?= $(firstword $(wildcard $(VENDORED_HEADERS)/libcuvs/lib64))
else
  # Cada pacote traz include/ e lib64/. RMM e rapids_logger têm parte compilada
  # (librmm.so, librapids_logger.so) que precisa entrar no link; RAFT só é necessário
  # se o linker reclamar de símbolos raft:: (ative com LINK_RAFT=1).
  # libcuvs vem como dependência do cuml-cu12 e traz a busca de vizinhança por epsilon,
  # que é o que o DBSCAN do cuML usa no caminho de força bruta.
  RAFT_PKG       := $(shell $(PYTHON) -c "import libraft, os; print(os.path.dirname(libraft.__file__))" 2>/dev/null)
  RMM_PKG        := $(shell $(PYTHON) -c "import librmm, os; print(os.path.dirname(librmm.__file__))" 2>/dev/null)
  LOGGER_PKG     := $(shell $(PYTHON) -c "import rapids_logger, os; print(os.path.dirname(rapids_logger.__file__))" 2>/dev/null)
  # Os cabeçalhos do cuVS podem vir no wheel C++ (libcuvs-cu12) ou junto do pacote Python
  # (cuvs-cu12), conforme a versão do RAPIDS. Tenta os dois e só aceita o diretório em que
  # o cabeçalho realmente existe.
  CUVS_PKG       := $(shell for m in libcuvs cuvs; do \
                      $(PYTHON) -c "import $$m, os; print(os.path.dirname($$m.__file__))" 2>/dev/null && break; \
                    done)
  RAFT_INCLUDE   ?= $(if $(RAFT_PKG),$(RAFT_PKG)/include,)
  RMM_INCLUDE    ?= $(if $(RMM_PKG),$(RMM_PKG)/include,)
  LOGGER_INCLUDE ?= $(if $(LOGGER_PKG),$(LOGGER_PKG)/include,)
  CUVS_INCLUDE   ?= $(patsubst %/cuvs/neighbors/epsilon_neighborhood.hpp,%,\
                      $(firstword $(wildcard $(CUVS_PKG)/include/cuvs/neighbors/epsilon_neighborhood.hpp)))
  RAFT_LIBDIR    ?= $(if $(RAFT_PKG),$(RAFT_PKG)/lib64,)
  RMM_LIBDIR     ?= $(if $(RMM_PKG),$(RMM_PKG)/lib64,)
  LOGGER_LIBDIR  ?= $(if $(LOGGER_PKG),$(LOGGER_PKG)/lib64,)
  CUVS_LIBDIR    ?= $(if $(CUVS_PKG),$(CUVS_PKG)/lib64,)
endif

# Backend da busca de vizinhança:
#   cuvs  (padrão) — cuvs::neighbors::epsilon_neighborhood::compute, exatamente a chamada
#                    que o DBSCAN do cuML faz no caminho de força bruta;
#   codes          — kernel próprio que grava o índice do menor eps por par, sem libcuvs.
# O backend cuvs é o que corresponde a "reaproveitar o cuML"; o codes existe como
# alternativa quando libcuvs não está disponível e como segunda implementação para
# validação cruzada.
BACKEND ?= cuvs

# CCCL (cuda/std, thrust, cub). O wheel do librmm traz, em include/rapids, exatamente a
# versão contra a qual RAFT e RMM foram compilados. Usá-la evita depender do CCCL que
# vem no CUDA Toolkit do módulo, que pode ser mais antigo. Como os -I explícitos são
# procurados antes dos diretórios do toolkit, este vence.
CCCL_INCLUDE ?= $(firstword $(wildcard $(RMM_INCLUDE)/rapids))

INCLUDES = -Isrc/compat \
           -Isrc \
           -Ithird_party/cuml/cpp/include \
           -Ithird_party/cuml/cpp/src/dbscan \
           $(if $(CCCL_INCLUDE),-I$(CCCL_INCLUDE),) \
           $(if $(RAFT_INCLUDE),-I$(RAFT_INCLUDE),) \
           $(if $(RMM_INCLUDE),-I$(RMM_INCLUDE),) \
           $(if $(LOGGER_INCLUDE),-I$(LOGGER_INCLUDE),) \
           $(if $(CUVS_INCLUDE),-I$(CUVS_INCLUDE),)

# Defines que o CMake do RMM/RAFT injeta automaticamente e que, num build com nvcc
# direto, precisam ser passados à mão. O RMM aborta com #error sem o primeiro.
DEFINES = -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
ifeq ($(BACKEND),cuvs)
  DEFINES += -DDBSCANMULTI_USE_CUVS
endif

NVCCFLAGS = -O3 -std=$(STD) -arch=$(CUDA_ARCH) \
            --expt-extended-lambda --expt-relaxed-constexpr $(DEFINES)

# RMM e rapids_logger não são header-only: rmm::cuda_stream_view, rmm::device_buffer,
# rmm::bad_alloc e afins vivem em librmm.so. O -rpath evita ter de mexer em
# LD_LIBRARY_PATH para rodar o binário depois.
# O nvcc não entende '-Wl,...'; opções de linker vão uma a uma por -Xlinker.
LDFLAGS =
ifneq ($(strip $(RMM_LIBDIR)),)
  LDFLAGS += -L$(RMM_LIBDIR) -lrmm -Xlinker -rpath -Xlinker $(RMM_LIBDIR)
endif
ifneq ($(strip $(LOGGER_LIBDIR)),)
  LDFLAGS += -L$(LOGGER_LIBDIR) -lrapids_logger -Xlinker -rpath -Xlinker $(LOGGER_LIBDIR)
endif
# libraft.so só entra se o linker reclamar de símbolos raft::. Fica opcional porque ela
# arrasta nccl/cublas/cusolver/cusparse e não é necessária para as partes header-only
# que este projeto usa.
ifeq ($(LINK_RAFT),1)
ifneq ($(strip $(RAFT_LIBDIR)),)
  LDFLAGS += -L$(RAFT_LIBDIR) -lraft -Xlinker -rpath -Xlinker $(RAFT_LIBDIR)
endif
endif
# libcuvs traz a busca de vizinhança por epsilon; entra só no backend cuvs.
ifeq ($(BACKEND),cuvs)
ifneq ($(strip $(CUVS_LIBDIR)),)
  LDFLAGS += -L$(CUVS_LIBDIR) -lcuvs -Xlinker -rpath -Xlinker $(CUVS_LIBDIR)
endif
endif

HEADERS = $(wildcard src/multi/*.cuh) $(wildcard src/compat/cuml/common/*.hpp)

.PHONY: all selftest check-headers clean

all: $(TARGET)

$(TARGET): src/main.cu $(HEADERS)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

selftest: $(TARGET)
	$(TARGET) --selftest --json

# Falha cedo e com mensagem clara se RAFT/RMM não estiverem no include path.
check-headers:
	@echo "BACKEND        = $(BACKEND)"
	@echo "RAFT_INCLUDE   = $(RAFT_INCLUDE)"
	@echo "RMM_INCLUDE    = $(RMM_INCLUDE)"
	@echo "LOGGER_INCLUDE = $(LOGGER_INCLUDE)"
	@echo "CUVS_INCLUDE   = $(if $(CUVS_INCLUDE),$(CUVS_INCLUDE),(não encontrado))"
	@echo "CCCL_INCLUDE   = $(if $(CCCL_INCLUDE),$(CCCL_INCLUDE),(nenhum: usando o CCCL do CUDA Toolkit))"
	@echo "RMM_LIBDIR     = $(RMM_LIBDIR)"
	@echo "LOGGER_LIBDIR  = $(LOGGER_LIBDIR)"
	@echo "RAFT_LIBDIR    = $(RAFT_LIBDIR) $(if $(filter 1,$(LINK_RAFT)),(linkado),(não linkado; use LINK_RAFT=1 se faltar símbolo raft::))"
	@echo "CUVS_LIBDIR    = $(CUVS_LIBDIR) $(if $(filter cuvs,$(BACKEND)),(linkado),(não linkado: BACKEND=$(BACKEND)))"
	@missing=0; cuvs_missing=0; \
	if [ "$(BACKEND)" = "cuvs" ]; then \
	  if [ -n "$(CUVS_INCLUDE)" ] && [ -f "$(CUVS_INCLUDE)/cuvs/neighbors/epsilon_neighborhood.hpp" ]; then \
	    echo "  ok      cuvs/neighbors/epsilon_neighborhood.hpp"; \
	  else \
	    echo "  FALTA   cuvs/neighbors/epsilon_neighborhood.hpp"; \
	    cuvs_missing=1; \
	  fi; \
	fi; \
	for h in raft/core/handle.hpp raft/core/error.hpp raft/sparse/csr.hpp \
	         raft/sparse/convert/csr.cuh raft/label/classlabels.cuh \
	         raft/label/merge_labels.cuh raft/util/cuda_dev_essentials.cuh \
	         raft/util/cudart_utils.hpp; do \
	  if [ -n "$(RAFT_INCLUDE)" ] && [ -f "$(RAFT_INCLUDE)/$$h" ]; then \
	    echo "  ok      $$h"; \
	  else \
	    echo "  FALTA   $$h"; missing=1; \
	  fi; \
	done; \
	for h in rmm/device_uvector.hpp rmm/cuda_stream_view.hpp; do \
	  if [ -n "$(RMM_INCLUDE)" ] && [ -f "$(RMM_INCLUDE)/$$h" ]; then \
	    echo "  ok      $$h"; \
	  else \
	    echo "  FALTA   $$h"; missing=1; \
	  fi; \
	done; \
	for h in rapids_logger/log_levels.h rapids_logger/logger.hpp; do \
	  if [ -n "$(LOGGER_INCLUDE)" ] && [ -f "$(LOGGER_INCLUDE)/$$h" ]; then \
	    echo "  ok      $$h"; \
	  else \
	    echo "  FALTA   $$h"; missing=1; \
	  fi; \
	done; \
	if [ "$$missing" = "1" ]; then \
	  echo ""; \
	  echo "RAFT/RMM/rapids_logger não encontrados. Opções, da mais simples para a mais pesada:"; \
	  echo "  1) bash scripts/fetch_rapids_headers.sh   (só curl + python3, sem venv)"; \
	  echo "  2) bash scripts/setup_env.sh              (venv completo)"; \
	  echo "  3) conda install -c rapidsai -c conda-forge libraft-headers librmm"; \
	  echo "     e compile com RAPIDS_INCLUDE=<prefixo>/include"; \
	fi; \
	if [ "$$cuvs_missing" = "1" ]; then \
	  echo ""; \
	  echo "cuVS ausente. É ele que faz a busca de vizinhança do DBSCAN do cuML, então o"; \
	  echo "backend padrão depende dele. Saídas:"; \
	  echo "  1) pip install --extra-index-url https://pypi.nvidia.com libcuvs-cu12"; \
	  echo "  2) bash scripts/fetch_rapids_headers.sh   (extrai o wheel, sem pip)"; \
	  echo "  3) make BACKEND=codes                     (kernel próprio, sem cuVS)"; \
	fi; \
	if [ "$$missing" = "1" ] || [ "$$cuvs_missing" = "1" ]; then exit 1; fi

clean:
	rm -rf $(BUILD_DIR)
