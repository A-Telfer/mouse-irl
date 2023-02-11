import numpy as np
import pandas as pd
import cv2

def transform_array_to_perspective(arr, T):
    """Move into the box's frame of reference"""
    x, y = arr.T
    tx, ty, v = T @ np.c_[x, y, np.ones_like(x)].T
    return np.c_[tx / v, ty / v]

def transform_dataframe_to_perspective(df, T):
    """Transform the coordinate dataframes to be in the box's frame of reference"""
    df = df.copy().dropna()
    idx = pd.IndexSlice
    x = df.loc[:, idx[:, :, "x"]]
    y = df.loc[:, idx[:, :, "y"]]
    x = x.stack(dropna=False).stack(dropna=False)
    y = y.stack(dropna=False).stack(dropna=False)

    tx, ty, v = T @ np.c_[x, y, np.ones_like(x)].T
    tx = tx / v
    ty = ty / v

    tx = pd.DataFrame(tx, index=x.index, columns=x.columns).unstack().unstack()
    ty = pd.DataFrame(ty, index=y.index, columns=y.columns).unstack().unstack()

    # Update multi index columns to match
    df.loc[:, pd.IndexSlice[:, :, "x"]] = tx
    df.loc[:, pd.IndexSlice[:, :, "y"]] = ty
    return df

def get_homography(points, target):
    T, res = cv2.findHomography(
        points, target, cv2.RANSAC, ransacReprojThreshold=32)
    return T