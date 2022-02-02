import math
import numpy as np
import pandas as pd

from .chisquare import chisquare


def generalize_target(x):
    return x - 10 if x > 10 else x


def generalize_others(x):
    if x == 2:
        return 1
    elif x == 12:
        return 11
    else:
        return x


def chi_square_pgm(X, Y, Z, df: pd.DataFrame):
    """
    Modification of Chi-square conditional independence test from pgmpy
    Tests the null hypothesis that X is independent from Y given Zs.
    """
    X = str(int(X))
    Y = str(int(Y))
    # X = int(X)
    # Y = int(Y)
    if isinstance(Z, (frozenset, list, set, tuple)):
        Z = list(Z)
    Z = [str(int(z)) for z in Z]

    state_names = {
        var_name: df.loc[:, var_name].unique() for var_name in df.columns
    }

    #print('State names', state_names)

    row_index = state_names[X]
    column_index = pd.MultiIndex.from_product(
            [state_names[Y]] + [state_names[z] for z in Z], names=[Y] + Z
        )

    XYZ_state_counts = pd.crosstab(
                index=df[X], columns= [df[Y]] + [df[z] for z in Z],
                rownames=[X], colnames=[Y] + Z
            )

    if not isinstance(XYZ_state_counts.columns, pd.MultiIndex):
            XYZ_state_counts.columns = pd.MultiIndex.from_arrays([XYZ_state_counts.columns])
    XYZ_state_counts = XYZ_state_counts.reindex(
            index=row_index, columns=column_index
        ).fillna(0)

    if Z:
        # Marginalize out Y
        XZ_state_counts = XYZ_state_counts.groupby(axis=1, level=list(range(1, len(Z)+1))).sum()
        # Marginalize out X
        YZ_state_counts = XYZ_state_counts.sum().unstack(Z)
    else:
        XZ_state_counts = XYZ_state_counts.sum(axis=1)
        YZ_state_counts = XYZ_state_counts.sum()
    # Marginalize out both
    Z_state_counts = YZ_state_counts.sum()

    XYZ_expected = np.zeros(XYZ_state_counts.shape)

    r_index = 0
    for X_val in XYZ_state_counts.index:
        X_val_array = []
        if Z:
            for Y_val in XYZ_state_counts.columns.levels[0]:
                temp = XZ_state_counts.loc[X_val] * YZ_state_counts.loc[Y_val] / Z_state_counts
                X_val_array = X_val_array + list(temp.to_numpy())
            XYZ_expected[r_index] = np.asarray(X_val_array)
            r_index=+1
        else:
            for Y_val in XYZ_state_counts.columns:
                temp = XZ_state_counts.loc[X_val] * YZ_state_counts.loc[Y_val] / Z_state_counts
                X_val_array = X_val_array + [temp]
            XYZ_expected[r_index] = np.asarray(X_val_array)
            r_index=+1

    observed = XYZ_state_counts.to_numpy().reshape(1,-1)
    expected = XYZ_expected.reshape(1,-1)
    observed, expected = zip(*((o, e) for o, e in zip(observed[0], expected[0])
                               if not (e == 0 or math.isnan(e) )))
    chi2, significance_level = chisquare(observed, expected)

    return chi2, significance_level
