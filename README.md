# Trabalho A2 - execução

## Conteúdo

- Código-fonte: `analise_cortes.py`, `modelagem.py` e `inspecionar_bases.py`.
- Bases: `dados_uci/Customer Churn.csv`, `spam.csv` e `fraud.csv`.
- Fontes e licenças: `FONTES_E_LICENCAS.md`.
- Relatório: `relatorio.md`; figuras em `resultados/`.

## Executar

Requer Python 3.11. Abra o terminal nesta pasta e execute:

```powershell
python -m pip install -r requirements.txt
python modelagem.py
```

O script treina três algoritmos por domínio, otimiza os hiperparâmetros no
treino por validação cruzada, seleciona modelo/cortes na validação e gera
métricas, curvas, histogramas e beeswarms SHAP do teste em `resultados/`.
Para executar apenas um domínio: `python modelagem.py --dominio churn`
(opções: `churn`, `spam`, `fraud`). A semente é 42.

`analise_cortes.py` pode ser importado por outros programas usando a função
`analisar_cortes(y_true, y_prob, t1, t2)`. A execução direta desse arquivo
produz apenas uma demonstração sintética, que não integra os resultados finais.

As versões utilizadas estão em `resultados/versoes.json`. Os CSVs originais
não são alterados pela execução. Os resultados foram verificados quanto a
separação das partições, reprodução das métricas, contagens por faixa e SHAP.

## Pendências obrigatórias antes da entrega

Confirmar a fonte/versão e licença do arquivo local `spam.csv`, conforme
detalhado em `FONTES_E_LICENCAS.md`, e identificar os integrantes no relatório.
