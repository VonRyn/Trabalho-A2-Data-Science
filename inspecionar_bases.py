"""Inspecao inicial sem dependencias externas; nao modifica os CSVs."""
import csv
import json
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
BASES = {
    "churn": ("dados/Customer Churn.csv", "Churn"),
    "spam": ("dados/spam.csv", "v1"),
    "fraud": ("dados/fraud.csv", "is_fraud"),
}


def ler_csv(caminho):
    bruto = caminho.read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            texto = bruto.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    # StringIO preserva quebras de linha dentro de campos entre aspas.
    from io import StringIO
    leitor = csv.reader(StringIO(texto))
    cabecalho = next(leitor)
    return cabecalho, list(leitor), encoding


def inspecionar():
    resumo = {}
    for nome, (arquivo, alvo) in BASES.items():
        colunas, linhas, encoding = ler_csv(RAIZ / arquivo)
        indice = colunas.index(alvo)
        classes = Counter(linha[indice] for linha in linhas)
        resumo[nome] = {
            "arquivo": arquivo, "encoding": encoding, "linhas": len(linhas),
            "colunas": colunas,
            "classes": {k: {"quantidade": v, "percentual": 100 * v / len(linhas)}
                        for k, v in sorted(classes.items())},
            "duplicatas_exatas": len(linhas) - len(set(map(tuple, linhas))),
            "ausencias": {coluna or f"sem_nome_{i}": sum(
                i >= len(linha) or not linha[i].strip() for linha in linhas)
                for i, coluna in enumerate(colunas)},
        }
        if nome == "spam":
            por_texto = {}
            for linha in linhas:
                por_texto.setdefault(linha[1], set()).add(linha[indice])
            resumo[nome]["textos_repetidos_excedentes"] = len(linhas) - len(por_texto)
            resumo[nome]["textos_com_rotulos_conflitantes"] = sum(
                len(rotulos) > 1 for rotulos in por_texto.values())
        if nome == "churn" and "CustomerId" in colunas:
            ids = [linha[colunas.index("CustomerId")] for linha in linhas]
            resumo[nome]["clientes_repetidos_excedentes"] = len(ids) - len(set(ids))
    return resumo


if __name__ == "__main__":
    resultado = inspecionar()
    destino = RAIZ / "resultados"
    destino.mkdir(exist_ok=True)
    (destino / "inspecao_bases.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
