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
- 3.150 clientes, alvo `Churn`, sendo 1 = cancelamento.
- Conforme a UCI, atributos resumem os primeiros nove meses; o alvo é medido
  ao final de doze meses, com intervalo de três meses entre a observação dos
  atributos e a definição do alvo.
- Transformações do experimento: remoção de duplicatas exatas, separação
  estratificada, imputação/codificação/escala dentro dos pipelines.

## Fraude: Waddah Ali, Kaggle

- Arquivo: `dados/fraud.csv`.
- Página: https://www.kaggle.com/datasets/waddahali/fraud-detection
- Título: Fraud Detection.
- Licença exibida na página durante a consulta: **CC BY-NC-SA 4.0**.
- Termos: https://creativecommons.org/licenses/by-nc-sa/4.0/
- 7.000 registros **sintéticos**.
- Alvo: `is_fraud`, sendo 1 = fraude.
- `hour_of_day` representa categorias 1 (manhã), 2 (tarde), 3 (noite),
  conforme a descrição; não é uma hora de 0 a 23.
- Atribuir crédito, vincular a licença e indicar modificações; uso não
  comercial. Se distribuir uma adaptação da base, observar a condição de
  compartilhar sob a mesma licença. O CSV original permanece intacto.

## Spam: fonte exata ainda pendente de confirmação

- Arquivo local: `dados/spam.csv`, 5.572 linhas, colunas `v1`, `v2` e três sem nome.
- Fonte declarada:
  https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification
- A página consultada identifica Ashfak Yeafi como publicador e exibe
  **Apache 2.0**, com link https://www.apache.org/licenses/LICENSE-2.0.
- Entretanto, a versão atual (3) oferece `email.csv`, com duas colunas
  (`label`/`text`), e tamanho diferente do arquivo fornecido. Não foi comprovado
  que o CSV local veio dessa versão ou que a licença exibida se aplica a ele.
- A licença Apache 2.0 exibida nessa página não foi atribuída ao arquivo local,
  pois sua correspondência com a versão publicada não foi confirmada.
- O pipeline mapeia ham=0/spam=1, reúne trechos das colunas extras e remove
  mensagens repetidas após normalizar caixa e espaços para a deduplicação.

## Justificativa da substituição da base bancária

`Churn Modeling.csv`, da página https://www.kaggle.com/datasets/santoshd3/bank-customers,
foi inicialmente considerado. A página exibe “Other (specified in description)”
e sua descrição não esclarece os termos. Por esse motivo, a base foi
substituída pela Iranian Churn da UCI. Os resultados apresentados para churn
correspondem exclusivamente à base da UCI.
