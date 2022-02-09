# contains statistic evaluations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def inspect_attributes(df: pd.DataFrame):
    df.hist(bins=30)
    plt.show()
