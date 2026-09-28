# Fontes e licenças das bases

Consulta realizada em 22/09/2026. Os arquivos de entrada não foram sobrescritos.
Os hashes dos CSVs usados estão em `resultados/sha256_bases.json`.

## Churn: Iranian Churn, UCI

- Arquivo: `dados/Customer Churn.csv`.
- Página: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
- DOI/citação: Iranian Churn [Dataset]. (2020). UCI Machine Learning Repository.
  https://doi.org/10.24432/C5JW3Z
- Licença declarada pela UCI: **CC BY 4.0**.
- Termos: https://creativecommons.org/licenses/by/4.0/
- Download original preservado em `iranian_churn_uci.zip`.
- 3.150 clientes, alvo `Churn`, sendo 1 = cancelamento.
- Conforme a UCI, atributos resumem os primeiros nove meses; o alvo é medido
  ao final de doze meses. A separação temporal entre atributos e alvo deve ser
  explicada no relatório.
- Transformações do experimento: remoção de duplicatas exatas, separação
  estratificada, imputação/codificação/escala dentro dos pipelines. Creditar a
  fonte e indicar essas transformações ao compartilhar os resultados/dados.

## Fraude: Waddah Ali, Kaggle

- Arquivo: `dados/fraud.csv`.
- Página: https://www.kaggle.com/datasets/waddahali/fraud-detection
- Título: Fraud Detection.
- Licença exibida na página durante a consulta: **CC BY-NC-SA 4.0**.
- Termos: https://creativecommons.org/licenses/by-nc-sa/4.0/
- 7.000 registros **sintéticos**; não descrever como transações bancárias reais.
- Alvo: `is_fraud`, sendo 1 = fraude.
- `hour_of_day` representa categorias 1 (manhã), 2 (tarde), 3 (noite),
  conforme a descrição; não é uma hora de 0 a 23.
- Atribuir crédito, vincular a licença e indicar modificações; uso não
  comercial. Se distribuir uma adaptação da base, observar a condição de
  compartilhar sob a mesma licença. O CSV original permanece intacto.

## Spam: fonte exata ainda pendente de confirmação

- Arquivo local: `dados/spam.csv`, 5.572 linhas, colunas `v1`, `v2` e três sem nome.
- Fonte informada pelo usuário:
  https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification
- A página consultada identifica Ashfak Yeafi como publicador e exibe
  **Apache 2.0**, com link https://www.apache.org/licenses/LICENSE-2.0.
- Entretanto, a versão atual (3) oferece `email.csv`, com duas colunas
  (`label`/`text`), e tamanho diferente do arquivo fornecido. Não foi comprovado
  que o CSV local veio dessa versão ou que a licença exibida se aplica a ele.
- Antes de entregar/redistribuir esse CSV, confirmar a versão/origem efetiva e
  guardar os termos aplicáveis. Não atribuir Apache 2.0 ao arquivo local apenas
  pela semelhança de assunto. A análise local já pode ser revisada.
- O pipeline mapeia ham=0/spam=1, reúne trechos das colunas extras e remove
  mensagens repetidas após normalizar caixa e espaços para a deduplicação.

## Base bancária anterior: fora da entrega proposta

`Churn Modeling.csv`, da página https://www.kaggle.com/datasets/santoshd3/bank-customers,
foi preservado. A página exibe “Other (specified in description)” e sua descrição
não esclarece os termos. Por decisão do usuário, foi substituída pela base da UCI.
A pasta `resultados/churn`, se presente, pertence à tentativa anterior com a base
bancária e **não deve ser usada no relatório nem entregue como resultado final**.
Os resultados válidos do novo churn ficam em `resultados/churn_uci`.
