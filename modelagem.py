"""Experimento reproduzivel da A2. Execute: python modelagem.py.

Modelos ajustados so no treino; hiperparametros por CV; vencedor e cortes na
validacao. O teste e consultado apenas depois de congeladas essas escolhas.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import os

RAIZ = Path(__file__).resolve().parent
# Suporte opcional a dependencias instaladas no diretorio do projeto.
if (RAIZ / ".deps_modelagem").is_dir():
    sys.path.insert(0, str(RAIZ / ".deps_modelagem"))
os.environ.setdefault("MPLCONFIGDIR", str(RAIZ / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from analise_cortes import analisar_cortes
from inspecionar_bases import BASES, inspecionar

SEMENTE = 42
# Hipoteses didaticas, nao requisitos reais de um banco/empresa.
# Denominadores: todos os positivos reais e todos os negativos reais.
POLITICAS = {
    "churn": {"max_fn": 0.15, "max_fp": 0.10},
    "spam": {"max_fn": 0.10, "max_fp": 0.005},
    "fraud": {"max_fn": 0.05, "max_fp": 0.01},
}


def salvar_json(objeto, caminho):
    caminho.write_text(json.dumps(objeto, ensure_ascii=False, indent=2,
                                 default=lambda x: x.item() if hasattr(x, "item") else str(x)),
                       encoding="utf-8")


def carregar(nome):
    arquivo, alvo = BASES[nome]
    df = pd.read_csv(RAIZ / arquivo, encoding="cp1252" if nome == "spam" else "utf-8-sig")
    linhas_originais = len(df)
    # O indice preservado permite rastrear cada registro no CSV original.
    if nome == "spam":
        if not df[alvo].isin(["ham", "spam"]).all() or df["v2"].isna().any():
            raise ValueError("SMS com texto ausente ou rotulo inesperado.")
        extras = [c for c in df if c.startswith("Unnamed:")]
        # Algumas mensagens foram fragmentadas em colunas sem nome na exportacao.
        # Preservar esses trechos, sem tratar fragmentos como atributos separados.
        df["texto"] = df[["v2"] + extras].apply(
            lambda linha: ",".join(str(v) for v in linha if pd.notna(v)), axis=1)
        df["chave"] = df["texto"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        if (df.groupby("chave")[alvo].nunique() > 1).any():
            raise ValueError("Ha mensagens iguais com rotulos conflitantes; revisar antes de dividir.")
        df = df.drop_duplicates("chave")
        X, y = df["texto"], df[alvo].map({"ham": 0, "spam": 1})
    else:
        if not df[alvo].isin([0, 1]).all():
            raise ValueError("Alvo precisa conter somente 0 e 1.")
        df = df.drop_duplicates()
        y = df[alvo].astype(int)
        X = df.drop(columns=[alvo])
    return X, y, {"linhas_originais": linhas_originais, "linhas_utilizadas": len(y),
                  "duplicatas_removidas": linhas_originais - len(y)}


def candidatos(nome, X):
    if nome == "spam":
        prep = TfidfVectorizer(max_features=3000, sublinear_tf=True, strip_accents="unicode")
        algoritmos = {
            "regressao_logistica": (LogisticRegression(max_iter=2000, random_state=SEMENTE),
                                   {"modelo__C": [1, 10], "prep__ngram_range": [(1, 1), (1, 2)]}),
            "naive_bayes": (MultinomialNB(), {"modelo__alpha": [0.1, 0.5, 1.0],
                                              "prep__ngram_range": [(1, 1), (1, 2)]}),
            "random_forest": (RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=SEMENTE),
                              {"modelo__max_depth": [None, 20], "modelo__min_samples_leaf": [1, 3]}),
        }
    else:
        categoricas = ["Tariff Plan", "Status"] if nome == "churn" else ["device_type", "store_type", "hour_of_day"]
        numericas = [c for c in X if c not in categoricas]
        prep = ColumnTransformer([
            ("num", Pipeline([("imputar", SimpleImputer(strategy="median", add_indicator=True)),
                              ("escala", StandardScaler())]), numericas),
            ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categoricas),
        ])
        algoritmos = {
            "regressao_logistica": (LogisticRegression(max_iter=2000, random_state=SEMENTE),
                                   {"modelo__C": [0.1, 1, 10], "modelo__class_weight": [None, "balanced"]}),
            "arvore_decisao": (DecisionTreeClassifier(random_state=SEMENTE),
                               {"modelo__max_depth": [3, 6, None], "modelo__min_samples_leaf": [5, 20]}),
            "random_forest": (RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=SEMENTE),
                              {"modelo__max_depth": [8, None], "modelo__min_samples_leaf": [1, 5]}),
        }
    from sklearn.base import clone
    return {n: (Pipeline([("prep", clone(prep)), ("modelo", m)]), g)
            for n, (m, g) in algoritmos.items()}


def probabilidade(modelo, X):
    return modelo.predict_proba(X)[:, list(modelo.classes_).index(1)]


def metricas(y, p):
    pred = p >= 0.5
    precisao, recall, _ = precision_recall_curve(y, p)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"limiar": 0.5, "acuracia": accuracy_score(y, pred),
            "precisao": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "auc_roc": roc_auc_score(y, p),
            "auc_pr_trapezoidal": auc(recall, precisao), "average_precision": average_precision_score(y, p),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def escolher_cortes(y, p, politica, destino):
    y = np.asarray(y)
    grade = np.unique(np.r_[0, 0.001, np.arange(0.01, 1, 0.01), 0.999, 1])
    opcoes = []
    for t1 in grade:
        negativos = p < t1
        fn = float(np.mean(negativos[y == 1]))
        for t2 in grade[grade > t1]:
            positivos = p >= t2
            fp = float(np.mean(positivos[y == 0]))
            opcoes.append({"t1": t1, "t2": t2, "fn_sobre_positivos": fn,
                           "fp_sobre_negativos": fp,
                           "cobertura_automatica": float(np.mean(negativos | positivos)),
                           "viavel": fn <= politica["max_fn"] and fp <= politica["max_fp"]})
    pd.DataFrame(opcoes).to_csv(destino / "busca_cortes_validacao.csv", index=False)
    viaveis = [o for o in opcoes if o["viavel"]]
    if not viaveis:
        raise ValueError("Nenhum par atende a politica na validacao. Revisar politica antes de acessar o teste.")
    # Desempate: menor soma dos erros, menor t1, maior t2.
    return max(viaveis, key=lambda o: (o["cobertura_automatica"],
               -o["fn_sobre_positivos"] - o["fp_sobre_negativos"], -o["t1"], o["t2"]))


def explicar(modelo, X_treino, X_teste, destino):
    import shap
    rng = np.random.default_rng(SEMENTE)
    ids_ref = rng.choice(len(X_treino), min(40, len(X_treino)), replace=False)
    ids_teste = rng.choice(len(X_teste), min(80, len(X_teste)), replace=False)
    prep, estimador = modelo.named_steps["prep"], modelo.named_steps["modelo"]
    ref = prep.transform(X_treino.iloc[ids_ref])
    amostra = prep.transform(X_teste.iloc[ids_teste])
    ref = ref.toarray() if hasattr(ref, "toarray") else np.asarray(ref)
    amostra = amostra.toarray() if hasattr(amostra, "toarray") else np.asarray(amostra)
    nomes = prep.get_feature_names_out()
    metodo = "LinearExplainer"
    if isinstance(estimador, (LogisticRegression, MultinomialNB)):
        if isinstance(estimador, LogisticRegression):
            coef, intercepto = estimador.coef_[0], estimador.intercept_[0]
        else:
            # Diferenca dos log-escores = log-odds posterior de classe 1.
            coef = estimador.feature_log_prob_[1] - estimador.feature_log_prob_[0]
            intercepto = estimador.class_log_prior_[1] - estimador.class_log_prior_[0]
        explicador = shap.LinearExplainer((coef, intercepto), ref)
        explicacao = explicador(amostra)
        saida = "log-odds da classe positiva; SHAP positivo aumenta a probabilidade da classe 1"
        esperado = amostra @ coef + intercepto
    else:
        metodo = "TreeExplainer"
        # As arvores do sklearn avaliam entradas em float32. Usar a mesma
        # precisao no SHAP evita divergencia em valores proximos aos cortes.
        ref = ref.astype(np.float32)
        amostra = amostra.astype(np.float32)
        explicador = shap.TreeExplainer(estimador, data=ref, model_output="probability",
                                       feature_perturbation="interventional")
        explicacao = explicador(amostra)
        if explicacao.values.ndim == 3:
            explicacao = explicacao[:, :, list(estimador.classes_).index(1)]
        saida = "probabilidade da classe positiva (1)"
        esperado = estimador.predict_proba(amostra)[:, list(estimador.classes_).index(1)]
        erro_arvore = np.max(np.abs(explicacao.base_values + explicacao.values.sum(axis=1) - esperado))
        if erro_arvore > 1e-4:
            # Fallback independente da representacao interna das arvores.
            # Explica predict_proba diretamente, sem tolerar valores incorretos.
            metodo = "PermutationExplainer (5 ciclos; TreeExplainer falhou na verificacao de aditividade)"
            predizer = lambda a: estimador.predict_proba(a)[:, list(estimador.classes_).index(1)]
            explicador = shap.PermutationExplainer(predizer, ref, seed=SEMENTE)
            explicacao = explicador(amostra, max_evals=5 * (2 * amostra.shape[1] + 1))
    explicacao.feature_names = list(nomes)
    erro = float(np.max(np.abs(explicacao.base_values + explicacao.values.sum(axis=1) - esperado)))
    if erro > 1e-4:
        raise ValueError(f"SHAP nao reconstruiu a saida explicada: erro={erro}")
    shap.plots.beeswarm(explicacao, max_display=15, show=False, plot_size=(11, 7))
    plt.title("SHAP - " + saida.split(";")[0])
    plt.savefig(destino / "shap_beeswarm_teste.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    ranking = pd.DataFrame({"caracteristica": nomes, "media_abs_shap": np.abs(explicacao.values).mean(axis=0)})
    ranking.sort_values("media_abs_shap", ascending=False).to_csv(destino / "shap_importancias.csv", index=False)
    np.savez_compressed(destino / "shap_valores.npz", valores=explicacao.values,
                        dados=amostra, base=explicacao.base_values, nomes=nomes.astype(str))
    info = {"saida": saida, "metodo": metodo, "amostra_teste": len(ids_teste), "referencia_treino": len(ids_ref),
            "indices_csv_teste": X_teste.index[ids_teste].tolist(),
            "indices_csv_referencia": X_treino.index[ids_ref].tolist(), "erro_aditividade_max": erro,
            "observacao": "Indices comecam em zero, sem contar cabecalho. Valores numericos foram transformados pelo pipeline."}
    salvar_json(info, destino / "shap_metodologia.json")
    return info


def executar(nome, raiz_saida):
    destino = raiz_saida / ("churn_uci" if nome == "churn" else nome)
    destino.mkdir(parents=True, exist_ok=True)
    X, y, limpeza = carregar(nome)
    X_dev, X_teste, y_dev, y_teste = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMENTE)
    X_treino, X_val, y_treino, y_val = train_test_split(X_dev, y_dev, test_size=0.25,
                                                     stratify=y_dev, random_state=SEMENTE)
    divisoes = []
    for conjunto, yy in [("treino", y_treino), ("validacao", y_val), ("teste", y_teste)]:
        for indice, rotulo in yy.items():
            divisoes.append({"indice_csv": indice, "conjunto": conjunto, "classe": int(rotulo)})
    pd.DataFrame(divisoes).to_csv(destino / "divisoes.csv", index=False)
    distribuicao = pd.DataFrame(divisoes).groupby(["conjunto", "classe"]).size().rename("quantidade").reset_index()
    distribuicao["percentual"] = 100 * distribuicao["quantidade"] / distribuicao.groupby("conjunto")["quantidade"].transform("sum")
    distribuicao.to_csv(destino / "distribuicao_classes.csv", index=False)
    salvar_json(limpeza, destino / "limpeza.json")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMENTE)
    modelos, validacao = {}, []
    for algoritmo, (pipe, grade) in candidatos(nome, X_treino).items():
        print(f"[{nome}] Ajustando {algoritmo}...", flush=True)
        busca = GridSearchCV(pipe, grade, scoring="average_precision", cv=cv, n_jobs=1,
                             error_score="raise", return_train_score=False)
        busca.fit(X_treino, y_treino)
        modelos[algoritmo] = busca.best_estimator_
        pd.DataFrame(busca.cv_results_).to_csv(destino / f"cv_{algoritmo}.csv", index=False)
        linha = {"modelo": algoritmo, "ap_cv": busca.best_score_, "hiperparametros": busca.best_params_,
                 **metricas(y_val, probabilidade(busca.best_estimator_, X_val))}
        validacao.append(linha)
        print(f"[{nome}] {algoritmo}: AP validacao={linha['average_precision']:.4f}", flush=True)
    pd.DataFrame(validacao).to_csv(destino / "comparacao_validacao.csv", index=False)
    vencedor = max(validacao, key=lambda r: r["average_precision"])
    modelo = modelos[vencedor["modelo"]]
    cortes = escolher_cortes(y_val, probabilidade(modelo, X_val), POLITICAS[nome], destino)
    # Salvo antes de quaisquer previsoes/metricas no teste.
    selecao = {"modelo": vencedor["modelo"], "hiperparametros": vencedor["hiperparametros"],
               "metrica_selecao": "average_precision na validacao", "politica_didatica": POLITICAS[nome],
               "cortes_validacao": cortes, "semente": SEMENTE, "limiar_metricas_binarias": 0.5}
    salvar_json(selecao, destino / "selecao_antes_do_teste.json")
    import joblib
    joblib.dump(modelo, destino / "melhor_modelo.joblib")
    fig, eixos = plt.subplots(1, 2, figsize=(12, 5))
    resultados, predicoes = [], pd.DataFrame({"indice_csv": y_teste.index, "y_true": y_teste.to_numpy()})
    for algoritmo, m in modelos.items():
        p = probabilidade(m, X_teste)
        resultados.append({"modelo": algoritmo, **metricas(y_teste, p)})
        predicoes[algoritmo] = p
        fpr, tpr, _ = roc_curve(y_teste, p)
        precisao, recall, _ = precision_recall_curve(y_teste, p)
        eixos[0].plot(fpr, tpr, label=algoritmo)
        eixos[1].plot(recall, precisao, label=algoritmo)
        disp = ConfusionMatrixDisplay(confusion_matrix(y_teste, p >= 0.5, labels=[0, 1]), display_labels=[0, 1])
        disp.plot(colorbar=False)
        plt.title(f"{nome} - {algoritmo} - teste (limiar 0,50)")
        disp.figure_.savefig(destino / f"matriz_{algoritmo}.png", dpi=150, bbox_inches="tight")
        plt.close(disp.figure_)
    eixos[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    eixos[1].axhline(y_teste.mean(), color="black", linestyle="--", label="Prevalencia positiva")
    for ax, titulo, xx, yy in [(eixos[0], "ROC", "Taxa de falsos positivos", "Recall"),
                               (eixos[1], "Precision-Recall", "Recall", "Precisao")]:
        ax.set(xlabel=xx, ylabel=yy, title=f"{nome} - {titulo} - teste", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destino / "curvas_teste.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(resultados).to_csv(destino / "comparacao_teste.csv", index=False)
    predicoes.to_csv(destino / "probabilidades_teste.csv", index=False)
    r = analisar_cortes(y_teste, predicoes[vencedor["modelo"]], cortes["t1"], cortes["t2"],
                       nome_classe_positiva=nome, titulo=f"{nome} - melhor modelo - teste",
                       caminho_para_salvar=str(destino / "histograma_teste.png"))
    plt.close(r.figura)
    faixas = [{"faixa": faixa, "quantidade": r.contagem_por_faixa[faixa],
               "percentual_populacao": r.percentual_por_faixa[faixa],
               "positivos": r.contagem_positivos_por_faixa[faixa],
               "negativos": r.contagem_negativos_por_faixa[faixa],
               "percentual_positivos_na_faixa": r.proporcao_positivos_por_faixa[faixa],
               "percentual_negativos_na_faixa": r.proporcao_negativos_por_faixa[faixa]}
              for faixa in r.contagem_por_faixa]
    pd.DataFrame(faixas).to_csv(destino / "faixas_teste.csv", index=False)
    erros = {"fn_percentual_positivos": r.proporcao_pos_classificados_como_neg,
             "denominador_fn": r.total_positivos, "fp_percentual_negativos": r.proporcao_neg_classificados_como_pos,
             "denominador_fp": r.total_negativos}
    salvar_json(erros, destino / "erros_automaticos_teste.json")
    print(f"[{nome}] Gerando SHAP para {vencedor['modelo']}...", flush=True)
    info_shap = explicar(modelo, X_treino, X_teste, destino)
    return {"dominio": nome, "limpeza": limpeza, "selecao": selecao, "teste": resultados,
            "faixas": faixas, "erros": erros, "shap": info_shap}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dominio", choices=["todos", *BASES], nargs="+", default=["todos"])
    args = parser.parse_args()
    saida = RAIZ / "resultados"
    saida.mkdir(exist_ok=True)
    salvar_json(inspecionar(), saida / "inspecao_bases.json")
    versoes = {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
               "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__}
    import shap
    versoes["shap"] = shap.__version__
    salvar_json(versoes, saida / "versoes.json")
    salvar_json({nome: hashlib.sha256((RAIZ / arq).read_bytes()).hexdigest()
                 for nome, (arq, _) in BASES.items()}, saida / "sha256_bases.json")
    for nome in BASES if "todos" in args.dominio else args.dominio:
        resumo = executar(nome, saida)
        pasta = saida / ("churn_uci" if nome == "churn" else nome)
        salvar_json(resumo, pasta / "resumo.json")
        print(f"[{nome}] Concluido. Arquivos em {pasta}", flush=True)


if __name__ == "__main__":
    main()
