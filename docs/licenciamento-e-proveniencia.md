# Licenciamento e proveniência

Última revisão: 2026-08-10.

Este documento registra o que é conhecido; não concede direitos que os autores ou fontes
de origem ainda não concederam. `NOTICE` é um registro de atribuição, não uma licença.

## Estado por componente

| Componente | Origem | Evidência | Estado para redistribuição |
|---|---|---|---|
| `third_party/cuml/` | RAPIDS cuML `v26.02.00`, commit `22b12c8c3e378f17f35107f7fb4ffd65a3dce534` | Apache-2.0, licença copiada e blobs registrados em `VENDORED.md` | **Resolvido para o vendorizado**, preservadas as condições Apache-2.0 |
| `src/multi/{corepoints_multi,runner_multi,vertexdeg_cuvs}.cuh` | Derivados dos arquivos cuML identificados nos cabeçalhos | SPDX/copyright NVIDIA, aviso de modificação UFV e entrada no `NOTICE` | **Identificado arquivo a arquivo** sob Apache-2.0 |
| Demais arquivos com SPDX próprio | Código novo que declara Apache-2.0 no próprio arquivo | Cabeçalho do arquivo | A declaração vale para aquele arquivo; não resolve a licença global |
| Repositório como um todo | Autores do projeto | Não existe `LICENSE` na raiz | **Não resolvido; bloqueia release pública** |
| F2 `Morphy999/DBSCANMultiE` | Repositório privado, sem licença declarada | `docs/fontes-primarias.md`, §3 | **Bloqueado** até autorização/licença ou demonstração de implementação independente |
| Geradores derivados do trabalho SSCAD/INF-494 | Grupo do projeto | Origem descrita no README e no módulo Python | **Revisão interna necessária** de autoria e autorização para redistribuição |
| F4, convenções do cluster | Infraestrutura local | Convenções descritas, sem inventário de trechos copiados | Confirmar se houve cópia textual de scripts |
| F5 CUDA-DClust+ | Comparador opcional | Nenhum código F5 está vendorizado nesta revisão | Fora do artefato; revisar licença antes de incluir |

## Bloqueio F2

O projeto declara compatibilidade com o contrato de linha de comando e influência do
protocolo de F2. Isso, sozinho, não permite concluir nem que houve cópia protegida nem que
todo o trabalho foi independente. Até uma revisão humana:

- não atribuir Apache-2.0 a material cuja única origem seja F2;
- não publicar uma release afirmando que “reutiliza o harness” de F2;
- classificar cada possível correspondência como interface, ideia, implementação
  independente ou código adaptado;
- para código adaptado, obter autorização escrita/licença ou reimplementar sem o trecho;
- guardar a decisão e a evidência fora do Git se contiverem material privado, registrando
  aqui apenas o resultado da revisão.

## Gate de release

Uma release pública só pode ser marcada como pronta quando:

1. os autores escolherem e adicionarem a licença raiz;
2. titulares/autores e afiliações no `CITATION.cff` forem confirmados;
3. F2 e os geradores internos tiverem decisão de proveniência documentada;
4. todos os arquivos derivados preservarem avisos e aparecerem no `NOTICE`;
5. `provenance/project-status.json` estiver com `publication_ready: true` e sem bloqueios;
6. `python scripts/check_repo_metadata.py --publication` terminar com código zero.

Não altere `project-status.json` apenas para fazer o gate passar: ele é um registro da
decisão humana, não a decisão em si.

## Alegações permitidas enquanto o gate está aberto

É correto afirmar que o subconjunto vendorizado identificado é Apache-2.0 e que arquivos
específicos declaram SPDX. Não é correto afirmar que “todo o projeto é Apache-2.0”, que não
há impedimento de redistribuição ou que a proveniência de F2 já foi resolvida.
