
"""metrics_analysis_full.py
Gera estatísticas para exatamente os atributos e métricas listados na tabela:
    Complexidade  -> AvgCyclomatic, SumCyclomatic, MaxNesting
    Coesão        -> PercentLackOfCohesion
    Acoplamento   -> CountClassCoupled
    Herança       -> MaxInheritanceTree
    Reusabilidade -> CountDeclMethodAll (RFC), CountDeclMethod (WMC)
    Polimorfismo  -> % herdado  = (RFC - WMC) / RFC * 100
    Tamanho       -> CountLineCode  (LOC)
    Facilidade    -> RatioCommentToCode

Os CSVs devem ser exportados do Understand no nível 'Class' para a coluna 'Kind'.

Uso básico:
    python metrics_analysis_full.py --main ddd-example-main.csv \
                                    --refac ddd-example-refatoracao.csv
Dependências:
    pip install pandas scipy
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

RAW_COLS = {
    'AvgCyclomatic',
    'SumCyclomatic',
    'MaxNesting',
    'PercentLackOfCohesion',
    'CountClassCoupled',
    'MaxInheritanceTree',
    'CountDeclMethodAll',   # RFC
    'CountDeclMethod',      # WMC
    'CountLineCode',
    'RatioCommentToCode',
}

def load_class_df(csv_path: Path, id_col: str, kind_col: str, kind_value: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if kind_col not in df.columns:
        raise ValueError(f"Coluna {kind_col} ausente em {csv_path}")
    df = df[df[kind_col] == kind_value]
    if df.empty:
        raise ValueError(f"Nenhuma linha com {kind_col}='{kind_value}' em {csv_path}")
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltam colunas {missing} em {csv_path}")
    return df.set_index(id_col)

def derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Percentual de métodos herdados
    df = df.copy()
    df['PercentInherited'] = np.where(
        df['CountDeclMethodAll'] > 0,
        (df['CountDeclMethodAll'] - df['CountDeclMethod']) / df['CountDeclMethodAll'] * 100,
        np.nan
    )
    return df

def paired_stats(df_main: pd.DataFrame, df_refac: pd.DataFrame, metrics: list) -> pd.DataFrame:
    joined = df_main.join(df_refac, how='inner', lsuffix='_main', rsuffix='_refac')
    if joined.empty:
        raise ValueError("Sem classes em comum entre os CSVs")
    rows = []
    for m in metrics:
        a = joined[f"{m}_main"]
        b = joined[f"{m}_refac"]
        delta_pct = np.nan if a.mean() == 0 else (b.mean() - a.mean()) / a.mean() * 100
        try:
            _, pval = wilcoxon(a, b, zero_method='zsplit')
        except ValueError:
            pval = np.nan
        rows.append({
            'metric': m,
            'mean_main': a.mean(), 'mean_refac': b.mean(),
            'median_main': a.median(), 'median_refac': b.median(),
            'std_main': a.std(), 'std_refac': b.std(),
            'pct_delta': delta_pct,
            'p_wilcoxon': pval,
            'n_classes': len(a)
        })
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', required=True)
    parser.add_argument('--refac', required=True)
    parser.add_argument('--idcol', default='Name')
    parser.add_argument('--kindcol', default='Kind')
    parser.add_argument('--kindvalue', default='Class')
    parser.add_argument('--out', default='metrics_full_summary.csv')
    args = parser.parse_args()

    df_main = load_class_df(Path(args.main), args.idcol, args.kindcol, args.kindvalue)
    df_refac = load_class_df(Path(args.refac), args.idcol, args.kindcol, args.kindvalue)

    df_main = derive_metrics(df_main)
    df_refac = derive_metrics(df_refac)

    METRIC_LIST = [
        'AvgCyclomatic', 'SumCyclomatic', 'MaxNesting',
        'PercentLackOfCohesion',
        'CountClassCoupled',
        'MaxInheritanceTree',
        'CountDeclMethodAll', 'CountDeclMethod',
        'PercentInherited',
        'CountLineCode',
        'RatioCommentToCode',
    ]

    summary = paired_stats(df_main, df_refac, METRIC_LIST)
    summary.to_csv(args.out, index=False)
    print(f"Resumo salvo em {args.out}\n")
    print(summary.to_string(index=False, float_format='%.4f'))

if __name__ == '__main__':
    main()
