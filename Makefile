# cuML-DBSCANMulti — build do executável CUDA
#
# O binário é compilado direto com nvcc, sem o CMake do cuML e sem libcuml++;
# BACKEND=cuvs liga somente a biblioteca de vizinhança libcuvs.
# Bastam os wheels libraft/librmm/libcuvs/rapids_logger e o CUDA Toolkit: deles saem os
# cabeçalhos quanto as .so do link (RMM não é header-only — rmm::cuda_stream_view,
# rmm::device_buffer e rmm::bad_alloc vivem em librmm.so).
#
# Ordem dos -I (importante): src/compat vem primeiro, para que o shim autocontido de
# <cuml/common/logger.hpp> seja encontrado antes do original vendorizado, que depende de
# um cabeçalho gerado pelo CMake do cuML que este build direto não produz.
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
LINK_RAFT ?= 0
PROJECT_ROOT := $(abspath $(dir $(firstword $(MAKEFILE_LIST))))

ifeq ($(OS),Windows_NT)
  NULL_DEVICE := NUL
else
  NULL_DEVICE := /dev/null
endif

# O executavel real pertence a uma configuracao de build. O caminho historico
# build/dbscan_multi continua sendo atualizado por `make`, mas e apenas um alias de
# compatibilidade; automacoes devem consultar `make -s print-target`.
BACKEND ?= cuvs
VALID_BACKENDS := cuvs codes

ifneq ($(filter $(BACKEND),$(VALID_BACKENDS)),$(BACKEND))
  $(error BACKEND invalido '$(BACKEND)'; use um de: $(VALID_BACKENDS))
endif
ifneq ($(words $(LINK_RAFT)),1)
  $(error LINK_RAFT deve ser um unico valor: 0 ou 1)
endif
ifneq ($(filter $(LINK_RAFT),0 1),$(LINK_RAFT))
  $(error LINK_RAFT invalido '$(LINK_RAFT)'; use 0 ou 1)
endif
ifneq ($(words $(CUDA_ARCH)),1)
  $(error CUDA_ARCH deve ser um unico valor sem espacos)
endif
ifneq ($(words $(STD)),1)
  $(error STD deve ser um unico valor sem espacos)
endif

CONFIG_ID     := backend-$(BACKEND)_arch-$(CUDA_ARCH)_std-$(STD)_raft-$(LINK_RAFT)
CONFIG_DIR    := $(BUILD_DIR)/$(CONFIG_ID)
override TARGET := $(CONFIG_DIR)/dbscan_multi
DEPFILE       := $(TARGET).d
CONFIG_STAMP  := $(CONFIG_DIR)/build-config.txt
COMPAT_TARGET ?= $(BUILD_DIR)/dbscan_multi
COMPAT_STAMP  := $(COMPAT_TARGET).config

# Identidade da revisao, em ordem de precedencia. O git e' OPCIONAL: o cluster recebe a
# arvore por rsync, sem .git, e nao deve depender dele para se identificar.
#
# Cada candidato e' resolvido ANTES do encadeamento, e nao dentro dele, para que um git que
# responde `--is-inside-work-tree = true` mas falha em `rev-parse HEAD` — .git parcial,
# HEAD nao nascido, safe.directory — caia para o proximo em vez de emitir string vazia.
# Foi exatamente o que barrou o job 4949: GIT_SHA='' passava pelo portao de identidade como
# valor ausente, sem nenhum fallback.
SOURCE_REVISION ?=
GIT_SHA_CANDIDATO  := $(strip $(shell git rev-parse --short=12 HEAD 2>$(NULL_DEVICE)))
TREE_SHA_CANDIDATO := $(strip $(shell $(PYTHON) scripts/source_tree_hash.py 2>$(NULL_DEVICE)))

ifneq ($(strip $(SOURCE_REVISION)),)
  GIT_SHA       := $(strip $(SOURCE_REVISION))
  GIT_DIRTY     := -1
  REVISION_KIND := provided
else ifneq ($(GIT_SHA_CANDIDATO),)
  GIT_SHA       := $(GIT_SHA_CANDIDATO)
  GIT_DIRTY     := $(if $(strip $(shell git status --porcelain --untracked-files=normal 2>$(NULL_DEVICE))),1,0)
  REVISION_KIND := git
else ifneq ($(TREE_SHA_CANDIDATO),)
  # Hash SHA-256 dos arquivos que definem a execucao (ver scripts/source_tree_hash.py).
  # Identifica a arvore exata que gerou o binario, que e' o que o portao quer saber; nao
  # aponta para um commit, e por isso git_dirty fica indefinido.
  GIT_SHA       := $(TREE_SHA_CANDIDATO)
  GIT_DIRTY     := -1
  REVISION_KIND := source-tree-sha256
else
  # Sem git e sem hash da arvore nao ha como identificar o binario, e o portao de
  # identidade vai barrar. Avisar aqui poupa descobrir isso so depois de compilar.
  $(warning revisao nao identificada: nem git nem '$(PYTHON) scripts/source_tree_hash.py' \
responderam. Confira se scripts/source_tree_hash.py foi sincronizado e se '$(PYTHON)' existe, \
ou passe SOURCE_REVISION=<id> explicitamente.)
  GIT_SHA       := unknown
  GIT_DIRTY     := -1
  REVISION_KIND := source-tree-sha256
endif
BUILD_ID := $(CONFIG_ID)_rev-$(GIT_SHA)$(if $(filter 1,$(GIT_DIRTY)),-dirty,)
BUILD_FLAGS_COMPACT := std=$(STD),arch=$(CUDA_ARCH),backend=$(BACKEND),link_raft=$(LINK_RAFT)

# Valores sem espacos mantem o escaping previsivel entre make, o shell e nvcc.
PROVENANCE_DEFINES := \
  -DDBSCANMULTI_GIT_SHA=\"$(GIT_SHA)\" \
  -DDBSCANMULTI_GIT_DIRTY=$(GIT_DIRTY) \
  -DDBSCANMULTI_REVISION_KIND=\"$(REVISION_KIND)\" \
  -DDBSCANMULTI_BUILD_ID=\"$(BUILD_ID)\" \
  -DDBSCANMULTI_CUDA_ARCH=\"$(CUDA_ARCH)\" \
  -DDBSCANMULTI_BUILD_BACKEND=\"$(BACKEND)\" \
  -DDBSCANMULTI_BUILD_FLAGS=\"$(BUILD_FLAGS_COMPACT)\"

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
  # Cada pacote traz include/ e lib64/. RMM e cuVS têm parte compilada
  # que precisa entrar no link; RAFT só é necessário
  # se o linker reclamar de símbolos raft:: (ative com LINK_RAFT=1).
  # libcuvs vem como dependência do cuml-cu12 e traz a busca de vizinhança por epsilon,
  # que é o que o DBSCAN do cuML usa no caminho de força bruta.
  RAFT_PKG       := $(shell $(PYTHON) -c "import libraft, os; print(os.path.dirname(libraft.__file__))" 2>$(NULL_DEVICE))
  RMM_PKG        := $(shell $(PYTHON) -c "import librmm, os; print(os.path.dirname(librmm.__file__))" 2>$(NULL_DEVICE))
  LOGGER_PKG     := $(shell $(PYTHON) -c "import rapids_logger, os; print(os.path.dirname(rapids_logger.__file__))" 2>$(NULL_DEVICE))
  # Os cabeçalhos do cuVS podem vir no wheel C++ (libcuvs-cu12) ou junto do pacote Python
  # (cuvs-cu12), conforme a versão do RAPIDS. Tenta os dois e só aceita o diretório em que
  # o cabeçalho realmente existe.
ifeq ($(BACKEND),cuvs)
  CUVS_PKG       := $(shell $(PYTHON) -c "import importlib.util as u, os; s=u.find_spec('libcuvs') or u.find_spec('cuvs'); print(os.path.dirname(s.origin) if s and s.origin else '')" 2>$(NULL_DEVICE))
endif
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
           $(if $(and $(filter cuvs,$(BACKEND)),$(CUVS_INCLUDE)),-I$(CUVS_INCLUDE),)

# Defines que o CMake do RMM/RAFT injeta automaticamente e que, num build com nvcc
# direto, precisam ser passados à mão. O RMM aborta com #error sem o primeiro.
DEFINES = -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
ifeq ($(BACKEND),cuvs)
  DEFINES += -DDBSCANMULTI_USE_CUVS
endif

NVCCFLAGS ?= -O3
COMPILE_FLAGS = $(NVCCFLAGS) -std=$(STD) -arch=$(CUDA_ARCH) \
                --expt-extended-lambda --expt-relaxed-constexpr \
                $(DEFINES) $(PROVENANCE_DEFINES)

# RMM não é header-only e os headers do RAFT dependem de rapids_logger; as duas bibliotecas
# entram no link. O shim elimina apenas o logger gerado pelo CMake do cuML.
# O -rpath evita ter de mexer em
# LD_LIBRARY_PATH para rodar o binário depois.
# O nvcc não entende '-Wl,...'; opções de linker vão uma a uma por -Xlinker.
LDFLAGS ?=
LINK_FLAGS = $(LDFLAGS)
ifneq ($(strip $(RMM_LIBDIR)),)
  LINK_FLAGS += -L$(RMM_LIBDIR) -lrmm -Xlinker -rpath -Xlinker $(RMM_LIBDIR)
endif
ifneq ($(strip $(LOGGER_LIBDIR)),)
  LINK_FLAGS += -L$(LOGGER_LIBDIR) -lrapids_logger -Xlinker -rpath -Xlinker $(LOGGER_LIBDIR)
endif
# libraft.so só entra se o linker reclamar de símbolos raft::. Fica opcional porque ela
# arrasta nccl/cublas/cusolver/cusparse e não é necessária para as partes header-only
# que este projeto usa.
ifeq ($(LINK_RAFT),1)
ifneq ($(strip $(RAFT_LIBDIR)),)
  LINK_FLAGS += -L$(RAFT_LIBDIR) -lraft -Xlinker -rpath -Xlinker $(RAFT_LIBDIR)
endif
endif
# libcuvs traz a busca de vizinhança por epsilon; entra só no backend cuvs.
ifeq ($(BACKEND),cuvs)
ifneq ($(strip $(CUVS_LIBDIR)),)
  LINK_FLAGS += -L$(CUVS_LIBDIR) -lcuvs -Xlinker -rpath -Xlinker $(CUVS_LIBDIR)
endif
endif

# Arquivos que o linker realmente escolherá. Além do gate de existência, eles entram no
# DAG para que uma atualização in-place do wheel/biblioteca force relink.
RMM_LIBRARY  := $(if $(RMM_LIBDIR),$(firstword $(wildcard $(RMM_LIBDIR)/librmm.so $(RMM_LIBDIR)/librmm.a)),)
LOGGER_LIBRARY := $(if $(LOGGER_LIBDIR),$(firstword $(wildcard $(LOGGER_LIBDIR)/librapids_logger.so $(LOGGER_LIBDIR)/librapids_logger.a)),)
RAFT_LIBRARY := $(if $(and $(filter 1,$(LINK_RAFT)),$(RAFT_LIBDIR)),$(firstword $(wildcard $(RAFT_LIBDIR)/libraft.so $(RAFT_LIBDIR)/libraft.a)),)
CUVS_LIBRARY := $(if $(and $(filter cuvs,$(BACKEND)),$(CUVS_LIBDIR)),$(firstword $(wildcard $(CUVS_LIBDIR)/libcuvs.so $(CUVS_LIBDIR)/libcuvs.a)),)
EXTERNAL_LIBS := $(RMM_LIBRARY) $(LOGGER_LIBRARY) $(RAFT_LIBRARY) $(CUVS_LIBRARY)

REPO_HEADERS := $(wildcard \
  src/multi/*.cuh \
  src/compat/cuml/common/*.hpp \
  third_party/cuml/cpp/include/cuml/cluster/*.hpp \
  third_party/cuml/cpp/include/cuml/common/*.hpp \
  third_party/cuml/cpp/src/dbscan/*.cuh \
  third_party/cuml/cpp/src/dbscan/*/*.cuh \
  third_party/cuml/cpp/src/dbscan/*/*.h)

.DEFAULT_GOAL := all
.DELETE_ON_ERROR:

.PHONY: all activate selftest check-binary-identity print-target print-config \
        check-toolchain check-headers check-libs check-build-env \
        check-config-matrix dry-run-matrix clean FORCE

all: $(TARGET) activate

$(CONFIG_DIR):
	@mkdir -p "$@"

FORCE:

# O arquivo so troca de mtime se os flags efetivos mudarem. Assim uma mudanca de
# include/lib/proveniencia recompila, mesmo quando os quatro eixos do diretorio sao iguais.
$(CONFIG_STAMP): FORCE | $(CONFIG_DIR)
	@tmp="$@.tmp.$$$$"; \
	{ printf '%s\n' \
	    "build_id=$(BUILD_ID)" \
	    "git_sha=$(GIT_SHA)" \
	    "git_dirty=$(GIT_DIRTY)" \
	    "revision_kind=$(REVISION_KIND)" \
	    "backend=$(BACKEND)" \
	    "cuda_arch=$(CUDA_ARCH)" \
	    "std=$(STD)" \
	    "link_raft=$(LINK_RAFT)" \
	    "nvcc=$(NVCC)" \
	    "nvcc_path=$$(command -v "$(firstword $(NVCC))" 2>$(NULL_DEVICE) || true)" \
	    "nvcc_version=$$($(NVCC) --version 2>&1 | tr '\n' ' ')" \
	    "nvccflags=$(NVCCFLAGS)" \
	    "compile_flags=$(COMPILE_FLAGS)" \
	    "includes=$(INCLUDES)" \
	    "external_libs=$(EXTERNAL_LIBS)" \
	    "ldflags=$(LDFLAGS)" \
	    "link_flags=$(LINK_FLAGS)"; \
	} > "$$tmp"; \
	if [ -r "$@" ] && cmp -s "$$tmp" "$@"; then \
	  rm -f "$$tmp"; \
	else \
	  mv -f "$$tmp" "$@"; \
	fi

# -MMD/-MP cobre os includes transitivos reais (inclusive RAFT/RMM/cuVS); a lista
# explicita acima cobre o primeiro build e permite que arquivos novos do repo entrem no DAG.
$(TARGET): src/main.cu $(REPO_HEADERS) Makefile $(CONFIG_STAMP) $(EXTERNAL_LIBS) | check-build-env
	@set -e; tmp="$@.tmp.$$$$"; dep="$(DEPFILE).tmp.$$$$"; \
	trap 'rm -f "$$tmp" "$$dep"' EXIT HUP INT TERM; \
	$(NVCC) $(COMPILE_FLAGS) $(INCLUDES) -MMD -MP -MF "$$dep" -MT "$@" \
	  "$<" -o "$$tmp" $(LINK_FLAGS); \
	mv -f "$$tmp" "$@"; \
	mv -f "$$dep" "$(DEPFILE)"

# Alias legado atomico. Os scripts de benchmark usam TARGET, nunca este arquivo mutavel.
activate: $(TARGET)
	@tmp="$(COMPAT_TARGET).tmp.$$$$"; meta="$(COMPAT_STAMP).tmp.$$$$"; \
	trap 'rm -f "$$tmp" "$$meta"' EXIT HUP INT TERM; \
	cp "$(TARGET)" "$$tmp"; \
	cp "$(CONFIG_STAMP)" "$$meta"; \
	mv -f "$$tmp" "$(COMPAT_TARGET)"; \
	mv -f "$$meta" "$(COMPAT_STAMP)"

selftest: check-binary-identity
	"$(TARGET)" --selftest --backend "$(BACKEND)" --json

check-binary-identity: $(TARGET)
	@$(PYTHON) scripts/check_binary_identity.py \
	  --binary "$(TARGET)" --backend "$(BACKEND)" --cuda-arch "$(CUDA_ARCH)"

print-target:
	@echo $(TARGET)

print-config:
	@echo CONFIG_ID=$(CONFIG_ID)
	@echo TARGET=$(TARGET)
	@echo COMPAT_TARGET=$(COMPAT_TARGET)
	@echo BACKEND=$(BACKEND)
	@echo CUDA_ARCH=$(CUDA_ARCH)
	@echo STD=$(STD)
	@echo LINK_RAFT=$(LINK_RAFT)
	@echo GIT_SHA=$(GIT_SHA)
	@echo GIT_DIRTY=$(GIT_DIRTY)
	@echo REVISION_KIND=$(REVISION_KIND)
	@echo BUILD_ID=$(BUILD_ID)

check-toolchain:
	@command -v "$(firstword $(NVCC))" >/dev/null 2>&1 || { \
	  echo "FALTA: compilador '$(firstword $(NVCC))' nao esta no PATH" >&2; exit 1; \
	}

# Falha cedo e lista todos os headers externos usados pelos caminhos compilados.
check-headers:
	@echo "BACKEND      = $(BACKEND)"
	@echo "TARGET       = $(TARGET)"
	@echo "RAFT_INCLUDE = $(RAFT_INCLUDE)"
	@echo "RMM_INCLUDE  = $(RMM_INCLUDE)"
	@echo "LOGGER_INCLUDE = $(LOGGER_INCLUDE)"
	@echo "CUVS_INCLUDE = $(if $(CUVS_INCLUDE),$(CUVS_INCLUDE),(nao encontrado))"
	@echo "CCCL_INCLUDE = $(if $(CCCL_INCLUDE),$(CCCL_INCLUDE),(toolkit CUDA))"
	@missing=0; \
	for h in raft/core/error.hpp raft/core/handle.hpp \
	         raft/label/classlabels.cuh raft/label/merge_labels.cuh \
	         raft/sparse/csr.hpp raft/sparse/convert/csr.cuh \
	         raft/util/cuda_dev_essentials.cuh raft/util/cuda_utils.cuh \
	         raft/util/cudart_utils.hpp; do \
	  if [ -n "$(RAFT_INCLUDE)" ] && [ -f "$(RAFT_INCLUDE)/$$h" ]; then \
	    echo "  ok      $$h"; \
	  else \
	    echo "  FALTA   $$h"; missing=1; \
	  fi; \
	done; \
	for h in rmm/cuda_stream_view.hpp rmm/device_uvector.hpp; do \
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
	if [ "$(BACKEND)" = cuvs ]; then \
	  h=raft/core/device_mdspan.hpp; \
	  if [ -n "$(RAFT_INCLUDE)" ] && [ -f "$(RAFT_INCLUDE)/$$h" ]; then \
	    echo "  ok      $$h"; \
	  else \
	    echo "  FALTA   $$h"; missing=1; \
	  fi; \
	  for h in cuvs/distance/distance.hpp cuvs/neighbors/epsilon_neighborhood.hpp; do \
	    if [ -n "$(CUVS_INCLUDE)" ] && [ -f "$(CUVS_INCLUDE)/$$h" ]; then \
	      echo "  ok      $$h"; \
	    else \
	      echo "  FALTA   $$h"; missing=1; \
	    fi; \
	  done; \
	fi; \
	if [ "$$missing" = 1 ]; then \
	  echo ""; \
	  echo "Dependencias de headers ausentes. Opcoes:"; \
	  echo "  1) bash scripts/fetch_rapids_headers.sh"; \
	  echo "  2) bash scripts/setup_env.sh"; \
	  echo "  3) informe RAPIDS_INCLUDE=<prefixo>/include"; \
	  if [ "$(BACKEND)" = cuvs ]; then \
	    echo "  4) use BACKEND=codes para compilar sem cuVS"; \
	  fi; \
	  exit 1; \
	fi

# Verifica exatamente as bibliotecas para as quais LDFLAGS emite -l.
check-libs:
	@missing=0; \
	check_lib() { \
	  label="$$1"; path="$$2"; \
	  if [ -n "$$path" ] && [ -f "$$path" ]; then \
	    echo "  ok      $$label: $$path"; \
	  else \
	    echo "  FALTA   $$label: biblioteca .so/.a nao encontrada"; missing=1; \
	  fi; \
	}; \
	check_lib RMM "$(RMM_LIBRARY)"; \
	check_lib rapids_logger "$(LOGGER_LIBRARY)"; \
	if [ "$(LINK_RAFT)" = 1 ]; then check_lib RAFT "$(RAFT_LIBRARY)"; fi; \
	if [ "$(BACKEND)" = cuvs ]; then check_lib cuVS "$(CUVS_LIBRARY)"; fi; \
	if [ "$$missing" = 1 ]; then \
	  echo "Biblioteca requerida pelo link nao encontrada." >&2; exit 1; \
	fi

check-build-env: check-toolchain check-headers check-libs

# Gates CPU e multiplataforma: nao executam nvcc nem consultam uma GPU.
check-config-matrix:
	@$(PYTHON) scripts/check_build_matrix.py --make "$(MAKE)"

dry-run-matrix:
	@$(PYTHON) scripts/check_build_matrix.py --make "$(MAKE)" --dry-run

-include $(DEPFILE)

clean:
	@resolved="$$($(PYTHON) -c 'import pathlib,sys; root=pathlib.Path(sys.argv[2]).resolve(); raw=pathlib.Path(sys.argv[1]); target=(raw if raw.is_absolute() else root/raw).resolve(); (target != root and root in target.parents) or sys.exit(2); print(target.as_posix())' "$(BUILD_DIR)" "$(PROJECT_ROOT)" 2>$(NULL_DEVICE))" || { \
	  echo "recusando remover BUILD_DIR fora do projeto: '$(BUILD_DIR)'" >&2; exit 1; \
	}; \
	rm -rf -- "$$resolved"
