#
# Import all dependencies
#
import os
import sys
import traceback
import numpy as np
import pandas as pd
from lsdb import read_hats
from dask.distributed import Client
from nested_pandas import NestedDtype



def create_dataset(df,
                    target,
                    label=None,
                    partition_info=None):
    
    """
    Create .CSV files for each partition of the dataframe (df).
    Each partition will be stored in a separate directory.
    Each partition will have a .CSV file containing the class distribution.
    The directory structure will be as follows:
    target/
        objects/
            partition_0.csv

    Parameters: 
    ---------------------------------------------------------
        df (DataFrame): contains the catalog file
        target (str): directory path for the files to be stored
        label (str): label associated with the catalog.
                        Provide this value only if the 'Class' column 
                        is missing in the dataframe (df). 
        
    """
    # If the input dataframe is empty, return it immediately.
    if df.empty:
        return df
    #
    # First partition_info in None
    # Start from the 2nd partition_info
    #
    if partition_info is not None:
        info_df = pd.DataFrame()
        LC_COLUMN = "lc"
        n_partition = partition_info['number']
        str_div = partition_info['division']
        #
        #
        #
        try: 
        
            if not os.path.exists(target):
                os.makedirs(target, exist_ok=True)
            #
            dest = os.path.join(target, "objects")
            os.makedirs(dest, exist_ok=True)
            #
            df = df.assign(**{LC_COLUMN: df[LC_COLUMN].astype(NestedDtype.from_pandas_arrow_dtype(df.dtypes[LC_COLUMN]))},)
            df = df.dropna(subset=['lc'])
            #
            #
            #
            if "Class" not in df.columns:
                if label: 
                    #
                    #
                    #
                    df['Class'] = label
                else:
                    raise AttributeError(f"\nException Raised: You must provide a class/label to this catalog."
                        f"\nThe 'Class' column couldn't be inferred from the catalog and the 'label'" 
                        f"\nparameter is {label} . Please provide a 'Class' column to the catalog or define" 
                        f"\nthe 'label' parameter.")
            
            #
            # Save the number of classes and their counts in a .CSV file
            #
            unique, counts = np.unique(df['Class'], return_counts=True)
            info_df['label'] = unique
            info_df['size'] = counts
            info_df['start_index'] = str_div
            info_df.to_csv(os.path.join(dest, f'partition_{n_partition}.csv'), index=False)
            print(f"\n[INFO] Created partition_{n_partition}.csv file.")
           

        except Exception:
            print(f"\n\n[Traceback]\n {traceback.format_exc()}\n")

        # Return the processed dataframe
        return df
            
def main(path_to_read=None, path_to_store=None, label=None):
    #
    # Read catalog
    #
    read_catalog = read_hats(path_to_read, )
    # Create an empty dataframe with the expected output structure for the 'meta' argument.
    # It should include any new columns you add in your function.
    meta_df = read_catalog._ddf.head(0)
    if "Class" not in meta_df.columns:
        meta_df['Class'] = pd.Series(dtype='object') # Or whatever dtype 'Class' will be

    #
    catalog_compute = read_catalog._ddf.map_partitions(create_dataset, 
                                                    target=path_to_store,
                                                    label=label,
                                                    meta=meta_df)

    with Client(n_workers=6, threads_per_worker=1, memory_limit="10GB") as client:
        catalog_compute.compute()
        client.close()


if __name__ == "__main__":
    #
    path_to_read = "../dataset/multi-class/hats/zubercal_vclassre"
    path_to_store = "../dataset/test"
    #
    main(path_to_read=path_to_read, path_to_store=path_to_store)

    