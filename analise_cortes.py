"""
Programa da Parte 1 - Analise de pontos de corte para classificacao binaria.

Reutilizavel para qualquer modelo/base que produza probabilidades estimadas
para a classe positiva. Gera o histograma (azul = todas as instancias,
vermelho = instancias cuja classe real e positiva) e calcula as metricas
das tres faixas de decisao definidas pelos cortes t1 e t2.

Uso tipico:
    from analise_cortes import analisar_cortes

    resultado = analisar_cortes(
        y_true=y_teste,
        y_prob=probabilidades_teste,          # saida de model.predict_proba(X)[:, 1]
        t1=0.30,
        t2=0.70,
        largura_intervalo=10,                 # em pontos percentuais
        nome_classe_positiva="e fraude?",
        titulo="Distribuicao de probabilidades - Deteccao de Fraude",
    )
"""

from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ResultadoAnaliseCortes:
    """Guarda todos os numeros usados no relatorio, alem da figura gerada."""

    percentual_por_faixa: dict
    contagem_por_faixa: dict
    contagem_positivos_por_faixa: dict
    contagem_negativos_por_faixa: dict
    proporcao_positivos_por_faixa: dict
    proporcao_negativos_por_faixa: dict
    proporcao_pos_classificados_como_neg: float
    denominador_pos_classificados_como_neg: int
    proporcao_neg_classificados_como_pos: float
    denominador_neg_classificados_como_pos: int
    total_instancias: int
    total_positivos: int
    total_negativos: int
    figura: plt.Figure = field(repr=False)


def analisar_cortes(
    y_true,
    y_prob,
    t1: float,
    t2: float,
    largura_intervalo: float = 10,
    nome_classe_positiva: str = "classe positiva",
    titulo: str = "Distribuicao de probabilidades estimadas",
    caminho_para_salvar: str | None = None,
) -> ResultadoAnaliseCortes:
    """
    Parametros
    ----------
    y_true : array-like de 0/1
        Rotulo real de cada instancia (1 = classe positiva).
    y_prob : array-like de float em [0, 1]
        Probabilidade estimada da classe positiva para cada instancia
        (independente do rotulo real).
    t1, t2 : float em [0, 1], com t1 < t2
        Pontos de corte que definem as tres faixas de decisao.
    largura_intervalo : float
        Largura de cada intervalo do histograma, em pontos percentuais
        (ex.: 10 -> intervalos de 0-10%, 10-20%, ..., 90-100%).
    nome_classe_positiva : str
        Descricao da classe positiva do problema (ex.: "e spam?",
        "e fraude?"), usada apenas para deixar o grafico/relatorio claros.
    caminho_para_salvar : str, opcional
        Se informado, salva a figura em disco (ex.: "grafico.png").

    Retorna
    -------
    ResultadoAnaliseCortes
        Estrutura com todas as metricas exigidas pelo enunciado e a figura.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("y_true e y_prob precisam ser vetores unidimensionais.")
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true e y_prob precisam ter o mesmo numero de instancias.")
    if y_true.size == 0:
        raise ValueError("O conjunto avaliado nao pode estar vazio.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true deve conter apenas rotulos 0 e 1.")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("y_prob deve conter probabilidades finitas entre 0 e 1.")
    y_true = y_true.astype(int)
    if not (0 <= t1 < t2 <= 1):
        raise ValueError("Os cortes devem satisfazer 0 <= t1 < t2 <= 1.")
    if not (0 < largura_intervalo <= 100):
        raise ValueError("largura_intervalo deve estar entre 0 e 100.")

    n_total = len(y_true)
    prob_pct = y_prob * 100.0
    mask_pos = y_true == 1
    n_pos = int(mask_pos.sum())
    n_neg = n_total - n_pos

    # ---- Histograma (cada cor normalizada pelo total do grupo representado) ----
    bordas = np.append(np.arange(0, 100, largura_intervalo), 100)

    contagem_todas, _ = np.histogram(prob_pct, bins=bordas)
    contagem_positivas, _ = np.histogram(prob_pct[mask_pos], bins=bordas)

    pct_todas = contagem_todas / n_total * 100
    pct_positivas = contagem_positivas / n_pos * 100 if n_pos > 0 else np.zeros_like(contagem_positivas, dtype=float)

    centros = (bordas[:-1] + bordas[1:]) / 2
    largura_barra = np.diff(bordas) * 0.9

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(centros, pct_todas, width=largura_barra, color="steelblue",
           alpha=0.85, label="Todas as instancias (total = 100%)", edgecolor="white")
    ax.bar(centros, pct_positivas, width=largura_barra * 0.6,
           color="crimson", alpha=0.85,
           label=f"Instancias reais positivas ({nome_classe_positiva}; total = 100%)" if n_pos > 0 else "Sem instancias reais positivas",
           edgecolor="white")

    ax.axvline(t1 * 100, color="black", linestyle="--", linewidth=1.5)
    ax.axvline(t2 * 100, color="black", linestyle="--", linewidth=1.5)
    ax.text(t1 * 100, ax.get_ylim()[1] * 0.95, f" t1={t1*100:.1f}%",
            rotation=90, va="top", ha="right", fontsize=9)
    ax.text(t2 * 100, ax.get_ylim()[1] * 0.95, f" t2={t2*100:.1f}%",
            rotation=90, va="top", ha="left", fontsize=9)

    ax.set_xlabel("Probabilidade estimada da classe positiva (%)")
    ax.set_ylabel("Percentual do total de cada grupo (%)")
    ax.set_xlim(0, 100)
    ax.set_title(titulo)
    ax.legend()
    fig.tight_layout()

    if caminho_para_salvar:
        fig.savefig(caminho_para_salvar, dpi=150)

    # ---- Metricas por faixa de decisao ----
    faixas = {
        "Classificacao negativa automatica": y_prob < t1,
        "Analise manual": (y_prob >= t1) & (y_prob < t2),
        "Classificacao positiva automatica": y_prob >= t2,
    }

    percentual_por_faixa, contagem_por_faixa = {}, {}
    contagem_positivos_por_faixa, contagem_negativos_por_faixa = {}, {}
    proporcao_positivos_por_faixa, proporcao_negativos_por_faixa = {}, {}

    for nome_faixa, mask_faixa in faixas.items():
        n_faixa = int(mask_faixa.sum())
        n_pos_faixa = int((mask_faixa & mask_pos).sum())
        n_neg_faixa = n_faixa - n_pos_faixa

        contagem_por_faixa[nome_faixa] = n_faixa
        contagem_positivos_por_faixa[nome_faixa] = n_pos_faixa
        contagem_negativos_por_faixa[nome_faixa] = n_neg_faixa
        percentual_por_faixa[nome_faixa] = n_faixa / n_total * 100
        proporcao_positivos_por_faixa[nome_faixa] = (
            n_pos_faixa / n_faixa * 100 if n_faixa > 0 else float("nan")
        )
        proporcao_negativos_por_faixa[nome_faixa] = (
            n_neg_faixa / n_faixa * 100 if n_faixa > 0 else float("nan")
        )

    # ---- Erros das decisoes automaticas (denominador = total da classe real) ----
    mask_auto_neg = y_prob < t1
    mask_auto_pos = y_prob >= t2

    n_pos_virou_neg = int((mask_pos & mask_auto_neg).sum())
    n_neg_virou_pos = int((~mask_pos & mask_auto_pos).sum())

    proporcao_pos_classificados_como_neg = (
        n_pos_virou_neg / n_pos * 100 if n_pos > 0 else float("nan")
    )
    proporcao_neg_classificados_como_pos = (
        n_neg_virou_pos / n_neg * 100 if n_neg > 0 else float("nan")
    )

    return ResultadoAnaliseCortes(
        percentual_por_faixa=percentual_por_faixa,
        contagem_por_faixa=contagem_por_faixa,
        contagem_positivos_por_faixa=contagem_positivos_por_faixa,
        contagem_negativos_por_faixa=contagem_negativos_por_faixa,
        proporcao_positivos_por_faixa=proporcao_positivos_por_faixa,
        proporcao_negativos_por_faixa=proporcao_negativos_por_faixa,
        proporcao_pos_classificados_como_neg=proporcao_pos_classificados_como_neg,
        denominador_pos_classificados_como_neg=n_pos,
        proporcao_neg_classificados_como_pos=proporcao_neg_classificados_como_pos,
        denominador_neg_classificados_como_pos=n_neg,
        total_instancias=n_total,
        total_positivos=n_pos,
        total_negativos=n_neg,
        figura=fig,
    )


def imprimir_relatorio(resultado: ResultadoAnaliseCortes) -> None:
    """Imprime um resumo textual pronto para colar/adaptar no relatorio."""
    r = resultado
    print(f"Total de instancias avaliadas: {r.total_instancias}")
    print(f"  - Positivas: {r.total_positivos} ({r.total_positivos/r.total_instancias*100:.1f}%)")
    print(f"  - Negativas: {r.total_negativos} ({r.total_negativos/r.total_instancias*100:.1f}%)\n")

    print("Faixas de decisao:")
    for nome in r.percentual_por_faixa:
        print(f"  {nome}:")
        print(f"    - {r.percentual_por_faixa[nome]:.1f}% da populacao "
              f"({r.contagem_por_faixa[nome]} instancias)")
        print(f"    - Quantidades: {r.contagem_positivos_por_faixa[nome]} positivas, "
              f"{r.contagem_negativos_por_faixa[nome]} negativas")
        print(f"    - Dentro da faixa: {r.proporcao_positivos_por_faixa[nome]:.1f}% positivas, "
              f"{r.proporcao_negativos_por_faixa[nome]:.1f}% negativas")

    print("\nErros das decisoes automaticas:")
    print(f"  - Positivos classificados automaticamente como negativos: "
          f"{r.proporcao_pos_classificados_como_neg:.1f}% "
          f"(denominador: {r.denominador_pos_classificados_como_neg} positivos reais)")
    print(f"  - Negativos classificados automaticamente como positivos: "
          f"{r.proporcao_neg_classificados_como_pos:.1f}% "
          f"(denominador: {r.denominador_neg_classificados_como_pos} negativos reais)")


if __name__ == "__main__":
    # Demonstracao com dados sinteticos - troquem por y_true/y_prob reais de cada dominio.
    rng = np.random.default_rng(42)
    n = 2000
    y_true_demo = rng.binomial(1, 0.15, size=n)
    y_prob_demo = np.where(
        y_true_demo == 1,
        np.clip(rng.beta(5, 2, size=n), 0, 1),
        np.clip(rng.beta(2, 5, size=n), 0, 1),
    )

    resultado = analisar_cortes(
        y_true=y_true_demo,
        y_prob=y_prob_demo,
        t1=0.30,
        t2=0.70,
        largura_intervalo=10,
        nome_classe_positiva="classe positiva (demo)",
        titulo="Demonstracao - dados sinteticos",
        caminho_para_salvar="demo_histograma.png",
    )
    imprimir_relatorio(resultado)
