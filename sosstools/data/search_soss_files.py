from astropy.io import fits
import pandas as pd
import os
from glob import glob
import fnmatch
from collections import Counter


def search_soss_files(
    csv_file="/ifs/jwst/wit/niriss/soss/uncal_data_index.csv", counts=None, **kwargs
):
    """
    Search the index of SOSS uncal files and return all rows of files that satisfy the criteria

    Parameters
    ----------
    csv_file: str
        The location of the CSV index of all SOSS uncal files (created with `index_soss_files` function)

    Returns
    -------
    DataFrame
        The filtered DataFrame of rows that sitisfy the criteria
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Initialize a boolean mask for filtering rows
    mask = pd.Series(True, index=df.index)

    print("Valid keywords:")
    print(list(df.columns))

    for keyword, value in kwargs.items():
        if isinstance(value, str):
            if "*" in value:
                mask &= df[keyword].apply(lambda x: fnmatch.fnmatch(str(x), value))
            else:
                mask &= df[keyword] == value
        else:
            mask &= df[keyword] == value

    result = df[mask]

    print("\nFound {}/{} files satisfying criteria:".format(len(result), len(df)))
    for keyword, value in kwargs.items():
        print(keyword, "=", value)

    if counts in result.columns:

        # Get unique values of the specified keyword from the DataFrame
        unique_values = result[counts].unique()

        # Count occurrences of each unique value
        value_counts = result[counts].value_counts()

        # Convert the value counts Series to a list of (value, count) tuples
        stats = [(value, count) for value, count in value_counts.items()]

        print("\n", pd.DataFrame(stats, columns=[counts, "Counts"]))

    return result
