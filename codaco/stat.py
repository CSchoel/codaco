# contains statistic evaluations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, probplot, jarque_bera, zscore
from typing import *

def inspect_attributes(df: pd.DataFrame, plot=True) -> pd.DataFrame:
    df.hist(bins=30)
    # 1. Choose only numeric columns
    numeric = df.filter(items=[c for c,d in zip(df.columns, df.dtypes) if d.kind in ['i', 'f']])
    # 2. Search for outliers
    index = numeric.apply(lambda x: np.abs(zscore(x))) > 3
    outliers = numeric[[np.any(r) for i,r in index.iterrows()]]
    print(outliers)
    # 1. check numeric columns for normality with shapiro-wilk or jarque-bera ...
    # NOTE shapiro-wilk will also reject very small deviations for large N (> 5000)
    # NOTE jarque-bera needs sample sizes > 2000
    if df.shape[0] > 2000:
        normality = numeric.apply(lambda x: pd.Series(jarque_bera(x), index=["statistic", "p-value"]))
    else:
        normality = numeric.apply(lambda x: pd.Series(shapiro(x), index=["statistic", "p-value"]))
    # ... and with QQ-plot
    qqdata = []
    for _, d in numeric.items():
        qqdata.append(probplot(d, fit=True))
    normality.join(pd.DataFrame({ 'qqr' : [qqr for (_, _), (_, _, qqr) in qqdata] }))
    # 2. Assess normality based on SW and QQ parameters
    print(normality.T)
    # plt.show()
    # plt.close()
    # plt.figure()
    # w = int(np.ceil(np.sqrt(len(numeric.columns))))
    # for i, (c, d) in enumerate(numeric.items()):
    #     ax = plt.subplot(w, w, i + 1)
    #     (osm, osr), (slope, intercept, r) = probplot(d, fit=True)
    # plt.show()
    # plt.close()
