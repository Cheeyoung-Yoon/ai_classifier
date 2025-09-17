# %%
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
class DataFrameParser:
    def __init__(self):
        pass
    
    

    # after 3rd row, define the types and get only the obj columns :
    @staticmethod
    def get_object_columns(df, start_row=5, threshold=1.0,normalize_numeric_strings: bool = True ):
        """`
        Get columns with object data type from a DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        
        Returns:
        list: A list of column names with object data type.
        """
        # df_part = df.iloc[start_row:]
        # object_columns = [col for col in df_part.columns if df_part[col].dropna().dtype == 'object']
        # # filtered_columns = [col for col in object_columns if col.startswith("Q") 
        # #                     or col.startswith("A") 
        # #                     or col.startswith("sq")
        # #                     or col.startswith("B")]
        # return object_columns
        
        def _clean_str_arr(s: pd.Series) -> pd.Series:
            # Only operate on strings; leave others as-is
            s = s.astype("string")
            # Trim
            s = s.str.strip()
            # Convert accounting negatives "(123)" -> "-123"

            s = s.str.replace(",", "", regex=False)
            s = s.str.replace("_", "", regex=False)
            s = s.str.replace(" ", "", regex=False)
            s = s.str.replace("-", "", regex=False)
            s = s.str.replace(":", "", regex=False)
            s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
            return s

        df_part = df.iloc[start_row:]
        n_rows = len(df_part)
        if n_rows == 0:
            return []

        keep = []
        for col in df_part.columns:
            s_full = df_part[col]
            # Consider only object or new pandas string dtype
            if not (is_object_dtype(s_full) or is_string_dtype(s_full)):
                continue

            total = len(s_full)
            null_count = s_full.isna().sum()

            # Prepare a Series for numeric coercion from non-null entries
            s_nonnull = s_full[~s_full.isna()]

            if normalize_numeric_strings:
                s_for_num = _clean_str_arr(s_nonnull)
            else:
                # Make sure we coerce as strings to handle mixed objects
                s_for_num = s_nonnull.astype("string")

            # Try to coerce to numeric (ints/floats). Non-convertible -> NaN
            num = pd.to_numeric(s_for_num, errors="coerce")

            # Identify integer-like entries (e.g., "10", "10.0", "(1,234)" -> -1234)
            # Note: num is float; an integer-like value has no fractional part.
            is_int_like = num.notna() #  & np.isfinite(num) & (np.floor(num) == num)

            int_like_count = int(is_int_like.sum())

            ratio_int_or_null = (int_like_count + null_count) / total


            # dt = pd.to_datetime(s_for_num, errors="coerce")
            # is_date_like = dt.notna()
            # date_like_count = int(is_date_like.sum())
            # ratio_date_or_null = (date_like_count + null_count) / total


            # If ≥ threshold are integers or null, then it's not an object/categorical column
            if ratio_int_or_null < threshold:
                keep.append(col)

        return keep

    @staticmethod
    def get_question_columns(df, q_num , grouped_dict):
        if isinstance(q_num, str):
            columns = grouped_dict.get(q_num  , [])
            r_df = df[columns]
        elif isinstance(q_num, list):
            columns = []
            for q in q_num:
                columns.extend(grouped_dict.get(q, []))
            r_df = df[columns]
            return r_df