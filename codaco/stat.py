# contains statistic evaluations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, probplot

def inspect_attributes(df: pd.DataFrame):
    df.hist(bins=30)
    print(df)
    numeric = df.filter(items=[c for c,d in zip(df.columns, df.dtypes) if d.kind in ['i', 'f']])
    normality = numeric.apply(lambda x: pd.Series(shapiro(x), index=["statistic", "p-value"]))
    print(normality)
    plt.show()
    plt.close()
    probplot(numeric)
    plt.show()
    plt.close()
