# contains statistic evaluations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, probplot, jarque_bera, zscore
from typing import *

def inspect_attributes(df: pd.DataFrame, plot=True, plot_dir="plots") -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 1. Choose only numeric columns
    numeric = df.filter(items=[c for c,d in zip(df.columns, df.dtypes) if d.kind in ['i', 'f']])
    # 2. Search for outliers (> 3σ)
    index = numeric.apply(lambda x: np.abs(zscore(x))) > 3
    outliers = numeric[index][[np.any(r) for i,r in index.iterrows()]]
    # 3. Remove outliers (since all normality tests are extremely sensitive to them)
    pruned = numeric.drop(index=outliers.index)
    # 4. check numeric columns for normality with shapiro-wilk or jarque-bera ...
    # NOTE shapiro-wilk will also reject very small deviations for large N (> 5000)
    # NOTE jarque-bera needs sample sizes > 2000
    if df.shape[0] > 2000:
        normality = pruned.apply(lambda x: pd.Series(jarque_bera(x), index=["statistic", "p-value"])).T
    else:
        normality = pruned.apply(lambda x: pd.Series(shapiro(x), index=["statistic", "p-value"])).T
    # ... and with QQ-plot
    qqdata = []
    for _, d in pruned.items():
        qqdata.append(probplot(d, fit=True))
    qqrcol = pd.DataFrame({ 'qqr' : [qqr for (_, _), (_, _, qqr) in qqdata] }, index=normality.index)
    normality = normality.join(qqrcol)
    # 5. Assess normality based on SW and QQ parameters
    maybenormal = (normality['p-value'] > 0.05) | (normality["qqr"] > 0.99)
    normalcol = pd.DataFrame({ 'distribution': [("normal" if n else "other") for n in maybenormal.values]}, index=normality.index)
    normality = normality.join(normalcol)
    # TODO check for uniform distributions
    # TODO plot
    # pruned.hist(bins=30)
    # plt.show()
    # plt.close()
    # plt.figure()
    # w = int(np.ceil(np.sqrt(len(numeric.columns))))
    # for i, (c, d) in enumerate(numeric.items()):
    #     ax = plt.subplot(w, w, i + 1)
    #     (osm, osr), (slope, intercept, r) = probplot(d, fit=True)
    # plt.show()
    # plt.close()
    return outliers, normality
