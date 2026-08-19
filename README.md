# Pipeline de genealogia acadêmica

Pipeline em Python para integrar relações de orientação do Mathematics
Genealogy Project (MGP) com autores e trabalhos do OpenAlex.

## Requisitos

- Python 3.10 ou superior;
- acesso à API do OpenAlex;
- `OPENALEX_API_KEY` configurada no ambiente;
- PyTorch, NumPy e pandas para a etapa de treinamento;
- acesso ao banco SQLite do MGP.

O banco utilizado pela pipeline deve ficar na raiz do projeto clonado:

```text
academic-genealogy-openalex-mlp/mgp.sqlite
```

O arquivo contém dados necessários para extrair pesquisadores, orientadores e
relações de orientação. Para solicitar acesso ao banco, entre em contato com
**c.massi@aluno.ufabc.edu.br**.

## Configuração

Clone o projeto e entre na raiz do repositório:

```bash
git clone URL_DO_REPOSITORIO academic-genealogy-openalex-mlp
cd academic-genealogy-openalex-mlp
```

Configure a chave da API somente na sessão atual do terminal:

```bash
export OPENALEX_API_KEY='sua-chave-do-openalex'
```

Não coloque a chave em arquivos `.py`, CSVs, no `README.md` ou no repositório
Git.

Instale as dependências disponíveis no projeto:

```bash
python3 -m pip install pandas numpy torch
```

Os scripts ficam no diretório `scripts/`, e os CSVs, modelos e métricas
gerados também são gravados nesse diretório. O banco `mgp.sqlite` não deve ser
versionado se houver restrições de distribuição; nesse caso, solicite o
arquivo pelo contato acima e coloque-o manualmente na raiz do projeto.

## Execução da pipeline

Execute os scripts na ordem abaixo.

### 1. Extrair pesquisadores e orientadores do MGP

O script recebe um pesquisador inicial (`seed`) e a quantidade de
pesquisadores a extrair (`limit`):

```bash
python3 scripts/extract_researchers.py 1 1000 \
  --db-path ./mgp.sqlite
```

Saída padrão:

```text
researchers_with_advisors.csv
```

Para selecionar outro ponto de partida ou outra quantidade de pesquisadores,
altere os argumentos `seed` e `limit`.

### 2. Identificar autores no OpenAlex

```bash
python3 scripts/identify_researchers_openalex.py
```

Essa etapa usa nomes, instituições e informações da área de matemática para
desambiguar autores. O cache evita repetir consultas já processadas.

Arquivos principais gerados:

- `researchers_with_open_alex_id.csv`;
- `researchers_with_advisors_and_open_alex_id.csv`.

### 3. Manter somente relações com os dois IDs

```bash
python3 scripts/extract_openalex_relationships.py
```

O resultado é acrescentado de forma incremental em:

```text
researchers_with_advisors_and_open_alex_id_complete.csv
```

Para sincronizar IDs que já estejam no cache:

```bash
python3 scripts/sync_openalex_relationship_ids.py
```

### 4. Buscar trabalhos compartilhados

A busca abaixo usa os dois IDs OpenAlex e procura trabalhos em geral, não
somente dissertações:

```bash
python3 scripts/extract_dissertation_relationships.py
```

O nome do script é legado; o comportamento atual consulta trabalhos de
qualquer tipo. O resultado padrão é:

```text
researchers_advisors_works.csv
```

Para consultar trabalhos usando apenas nomes, sem os IDs OpenAlex, também
existe:

```bash
python3 scripts/extract_works.py
```

Essa alternativa gera `researchers_openalex_works.csv` e deve ser usada quando
o arquivo de entrada contiver as colunas `researcher_name` e `advisor_name`.

### 5. Filtrar linhas com trabalhos encontrados

```bash
python3 scripts/extract_works_found.py
```

Entrada padrão:

```text
researchers_advisors_works.csv
```

Saída padrão:

```text
researchers_advisors_works_found.csv
```

### 6. Treinar a MLP

```bash
python3 scripts/train_relationship_mlp.py
```

O treinamento usa 80% dos exemplos para treinamento e 20% para teste,
gera pares negativos amostrados e salva:

- `relationship_training_dataset.csv`: características e rótulos;
- `relationship_predictions.csv`: probabilidades e classes previstas;
- `relationship_test_metrics.json`: métricas do experimento;
- `relationship_mlp.pt`: pesos e parâmetros do modelo.

Parâmetros opcionais:

```bash
python3 scripts/train_relationship_mlp.py \
  --epochs 300 \
  --negative-ratio 1.0 \
  --seed 42
```

## Execução com caminhos personalizados

Todos os scripts que leem ou escrevem CSV aceitam opções `--input` e
`--output`. Por exemplo:

```bash
python3 scripts/extract_works_found.py \
  --input ./scripts/pesquisadores_works.csv \
  --output ./scripts/pesquisadores_works_found.csv
```

Consulte os demais parâmetros com:

```bash
python3 scripts/nome_do_script.py --help
```

## Observações importantes

- A execução é incremental em várias etapas e evita repetir relações já
  processadas.
- A ausência de um autor ou trabalho no OpenAlex não prova que a relação
  acadêmica inexiste.
- O OpenAlex não registra explicitamente relações de orientação.
- Os resultados do MLP são experimentais e dependem da qualidade da
  desambiguação, da cobertura do OpenAlex e do conjunto de exemplos.
- A API do OpenAlex possui limites de requisições; evite executar consultas
  repetidas sem necessidade.
