"""
Data utility functions for handling saved data files efficiently.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

class DataHelper:
    """Helper class for efficient data handling without loading into memory."""
    
    @staticmethod
    def load_dataframe(csv_path: str) -> pd.DataFrame:
        """Load DataFrame from saved CSV file."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        return pd.read_csv(csv_path, encoding='utf-8-sig', index_col=0)
    
    @staticmethod
    def get_dataframe_info(csv_path: str) -> Dict[str, Any]:
        """Get basic info about DataFrame without loading it into memory."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # Read only the first few rows to get column info
        sample_df = pd.read_csv(csv_path, encoding='utf-8-sig', nrows=5)
        
        return {
            "file_path": csv_path,
            "columns": sample_df.columns.tolist(),
            "shape_estimated": {"rows": "unknown", "cols": len(sample_df.columns)},
            "dtypes": sample_df.dtypes.to_dict(),
            "sample_data": sample_df.head(3).to_dict()
        }
    
    @staticmethod
    def get_columns(csv_path: str) -> List[str]:
        """Get column names without loading full DataFrame."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        # Read only header
        sample_df = pd.read_csv(csv_path, encoding='utf-8-sig', nrows=0)
        return sample_df.columns.tolist()
    
    @staticmethod
    def get_sample_data(csv_path: str, n_rows: int = 3, columns: Optional[List[str]] = None) -> Dict:
        """Get sample data without loading full DataFrame."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        if columns:
            # First check what columns actually exist
            sample_check = pd.read_csv(csv_path, encoding='utf-8-sig', nrows=0)
            existing_cols = sample_check.columns.tolist()
            
            # Filter to only existing columns
            valid_columns = [col for col in columns if col in existing_cols]
            if valid_columns:
                sample_df = pd.read_csv(csv_path, encoding='utf-8-sig', nrows=n_rows, usecols=valid_columns)
            else:
                # If no valid columns, just read first few columns
                sample_df = pd.read_csv(csv_path, encoding='utf-8-sig', nrows=n_rows)
        else:
            sample_df = pd.read_csv(csv_path, encoding='utf-8-sig', nrows=n_rows)
        
        return sample_df.to_dict()
    
    @staticmethod
    def get_open_columns_from_path(csv_path: str, meta_info: Dict[str, Any]) -> List[str]:
        """Get open-ended columns without loading full DataFrame."""
        if 'object_columns' in meta_info:
            return meta_info['object_columns']
        
        # Fallback: analyze column types from sample
        sample_df = pd.read_csv(csv_path, encoding='utf-8-sig', nrows=10)
        object_cols = sample_df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Filter out common non-text columns
        drop_names = ['START_TIME','END_TIME','DATA_TIME','ACCESS_KEY',
                     'USER_AGENT','USER_DEVICE','IS_MOBILE','IP','RESULT',
                     'LAST_QUESTION','LAST_BEFORE_QUESTION','면접원']
        
        return [col for col in object_cols if col not in drop_names]

class StateDataManager:
    """Manages data paths in state and provides lazy loading utilities."""
    
    @staticmethod
    def save_state_data(state: Dict[str, Any], data_key: str, save_path: str) -> Dict[str, Any]:
        """Save any data to file and store path in state."""
        # This is a placeholder for saving various types of data
        # Can be extended based on data types
        pass
    
    @staticmethod
    def load_state_data(state: Dict[str, Any], data_key: str, helper_class=None):
        """Load data from path stored in state."""
        if f"{data_key}_path" not in state:
            raise KeyError(f"No path found for {data_key}")
        
        path = state[f"{data_key}_path"]
        
        if helper_class:
            return helper_class.load_dataframe(path)
        else:
            # Generic file loading
            if path.endswith('.csv'):
                return pd.read_csv(path, encoding='utf-8-sig')
            elif path.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
