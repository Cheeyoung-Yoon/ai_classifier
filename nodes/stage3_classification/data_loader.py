"""
Simple data loader for Stage 3 MCL clustering.
Reads data from LangGraph state and loads CSV embeddings.
"""
import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from .config import Stage3Config
except ImportError:
    # Direct execution fallback
    from config import Stage3Config


def map_clusters_back_to_data(cluster_labels: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Map cluster labels back to original data structure with proper IDs.
    
    Args:
        cluster_labels: Cluster assignments for each embedding
        metadata: Metadata from load_data_from_state()
        
    Returns:
        Dictionary with detailed mapping information
    """
    if len(cluster_labels) != len(metadata["embedding_ids"]):
        raise ValueError(f"Cluster labels length {len(cluster_labels)} doesn't match embeddings {len(metadata['embedding_ids'])}")
    
    # Create mapping from embedding ID to cluster
    embedding_to_cluster = {}
    for i, (embedding_id, cluster_label) in enumerate(zip(metadata["embedding_ids"], cluster_labels)):
        embedding_to_cluster[embedding_id] = int(cluster_label)
    
    # Group by original respondent/row
    respondent_clusters = {}
    original_df = metadata["original_dataframe"]
    
    if metadata["format"] == "long":
        # Long format: multiple embeddings per row
        row_mapping = metadata["row_mapping"]
        col_mapping = metadata["col_mapping"]
        embed_cols = metadata["embedding_columns"]
        
        for i, (row_idx, col_idx) in enumerate(zip(row_mapping, col_mapping)):
            row = original_df.iloc[row_idx]
            respondent_id = row.get('respondent_id', row.get('id', f'row_{row_idx}'))
            col_name = embed_cols[col_idx]
            
            if respondent_id not in respondent_clusters:
                respondent_clusters[respondent_id] = {
                    "respondent_data": row.to_dict(),
                    "embedding_clusters": {},
                    "question_id": row.get('question_id', 'unknown')
                }
            
            respondent_clusters[respondent_id]["embedding_clusters"][col_name] = int(cluster_labels[i])
    
    else:
        # Single format: one embedding per row
        for i, embedding_id in enumerate(metadata["embedding_ids"]):
            row_idx = i  # In single format, index matches row
            row = original_df.iloc[row_idx]
            respondent_id = row.get('respondent_id', row.get('id', f'row_{row_idx}'))
            
            respondent_clusters[respondent_id] = {
                "respondent_data": row.to_dict(),
                "cluster": int(cluster_labels[i]),
                "question_id": row.get('question_id', 'unknown')
            }
    
    # Create summary statistics
    unique_clusters = np.unique(cluster_labels)
    cluster_summary = {}
    
    for cluster_id in unique_clusters:
        cluster_embeddings = [metadata["embedding_ids"][i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_summary[int(cluster_id)] = {
            "size": len(cluster_embeddings),
            "embedding_ids": cluster_embeddings
        }
    
    return {
        "respondent_clusters": respondent_clusters,
        "embedding_to_cluster": embedding_to_cluster,
        "cluster_summary": cluster_summary,
        "total_respondents": len(respondent_clusters),
        "total_clusters": len(unique_clusters),
        "format": metadata["format"]
    }


def load_data_from_state(state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load embeddings from LangGraph state for MCL clustering.
    
    Args:
        state: LangGraph state containing matched_questions data
        
    Returns:
        Tuple of (embeddings_array, metadata_dict)
    """
    # Get configuration
    config = Stage3Config.merge_with_state(state)
    use_long_format = config["use_long_format"]
    embedding_columns = config["embedding_columns"]
    
    # Get data from state
    matched_questions = state.get("matched_questions", {})
    if not matched_questions:
        raise ValueError("No matched_questions found in state")
    
    # Load dataframe from state
    df = _load_dataframe_from_state(matched_questions)
    
    print(f"📊 Loaded dataframe: {len(df)} rows, columns: {list(df.columns)}")
    
    # Extract embeddings
    if use_long_format:
        embeddings, metadata = _extract_long_format_embeddings(df, embedding_columns)
    else:
        embeddings, metadata = _extract_single_embeddings(df, embedding_columns)
    
    print(f"🎯 Extracted embeddings: {embeddings.shape}")
    
    return embeddings, metadata


def _load_dataframe_from_state(matched_questions: Dict[str, Any]) -> pd.DataFrame:
    """Load DataFrame from matched_questions in state.
    
    Args:
        matched_questions: matched_questions section of state
        
    Returns:
        Combined DataFrame
    """
    # Check for direct dataframe in state (test/mock pattern)
    if "dataframe" in matched_questions:
        data = matched_questions["dataframe"]
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            return data
        else:
            raise ValueError(f"Invalid dataframe format: {type(data)}")
    
    # Real state structure: matched_questions -> QID -> stage2_data -> csv_path
    dfs = []
    for question_id, question_data in matched_questions.items():
        if not isinstance(question_data, dict):
            continue
        
        # Check if this question has stage2_data
        stage2_data = question_data.get("stage2_data")
        if not stage2_data or not isinstance(stage2_data, dict):
            print(f"⚠️  Skipping {question_id}: No stage2_data found")
            continue
            
        # Check if processing is completed
        if stage2_data.get("status") != "completed":
            print(f"⚠️  Skipping {question_id}: Status is {stage2_data.get('status')}")
            continue
        
        # Get CSV path
        csv_path = stage2_data.get("csv_path")
        if not csv_path:
            print(f"⚠️  Skipping {question_id}: No csv_path found")
            continue
        
        # Load the CSV file
        try:
            df = _load_csv_file(csv_path)
            if df.empty:
                print(f"⚠️  Skipping {question_id}: CSV file is empty")
                continue
                
            # Add metadata columns
            df['question_id'] = question_id
            df['processing_type'] = stage2_data.get('processing_type', 'unknown')
            df['timestamp'] = stage2_data.get('timestamp', 'unknown')
            
            dfs.append(df)
            print(f"✅ Loaded {question_id}: {len(df)} rows from {Path(csv_path).name}")
            
        except Exception as e:
            print(f"❌ Failed to load {question_id}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid stage2 data found in any questions")
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"📊 Combined data: {len(dfs)} questions, {len(combined_df)} total rows")
    
    return combined_df


def _load_csv_file(file_path: str) -> pd.DataFrame:
    """Load a single CSV file with error handling.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Loaded DataFrame
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to load CSV {path}: {e}")


def _extract_long_format_embeddings(df: pd.DataFrame, embedding_columns: List[str]) -> Tuple[np.ndarray, Dict]:
    """Extract embeddings in long format (multiple columns combined).
    
    Args:
        df: Input DataFrame
        embedding_columns: List of potential embedding column name patterns
        
    Returns:
        Tuple of (embeddings_array, metadata)
    """
    # Find embedding columns
    embed_cols = []
    for col in df.columns:
        for pattern in embedding_columns:
            if pattern.lower() in col.lower():
                embed_cols.append(col)
                break
    
    if not embed_cols:
        raise ValueError(f"No embedding columns found. Looking for patterns: {embedding_columns}")
    
    print(f"🔍 Found embedding columns: {embed_cols}")
    
    # Collect all embeddings with IDs
    all_embeddings = []
    row_mapping = []
    col_mapping = []
    embedding_ids = []  # Track unique IDs for each embedding
    
    for row_idx, row in df.iterrows():
        for col_idx, col_name in enumerate(embed_cols):
            embedding = _parse_embedding(row[col_name])
            if embedding is not None and len(embedding) > 0:
                all_embeddings.append(embedding)
                row_mapping.append(row_idx)
                col_mapping.append(col_idx)
                
                # Create unique ID: row_id + column_name
                row_id = row.get('respondent_id', row.get('id', f'row_{row_idx}'))
                embedding_id = f"{row_id}_{col_name}"
                embedding_ids.append(embedding_id)
    
    if not all_embeddings:
        raise ValueError("No valid embeddings found in data")
    
    embeddings_array = np.array(all_embeddings)
    
    metadata = {
        "format": "long",
        "original_rows": len(df),
        "embedding_columns": embed_cols,
        "total_embeddings": len(all_embeddings),
        "row_mapping": np.array(row_mapping),
        "col_mapping": np.array(col_mapping),
        "embedding_ids": embedding_ids,  # IDs for mapping back
        "original_dataframe": df  # Keep reference for mapping
    }
    
    return embeddings_array, metadata


def _extract_single_embeddings(df: pd.DataFrame, embedding_columns: List[str]) -> Tuple[np.ndarray, Dict]:
    """Extract embeddings from a single column (original format).
    
    Args:
        df: Input DataFrame
        embedding_columns: List of potential embedding column name patterns
        
    Returns:
        Tuple of (embeddings_array, metadata)
    """
    # Find first embedding column
    embed_col = None
    for col in df.columns:
        for pattern in embedding_columns:
            if pattern.lower() in col.lower():
                embed_col = col
                break
        if embed_col:
            break
    
    if not embed_col:
        raise ValueError(f"No embedding column found. Looking for patterns: {embedding_columns}")
    
    print(f"🎯 Using single embedding column: {embed_col}")
    
    # Extract embeddings with IDs
    embeddings = []
    embedding_ids = []
    
    for row_idx, row in df.iterrows():
        embedding = _parse_embedding(row[embed_col])
        if embedding is not None and len(embedding) > 0:
            embeddings.append(embedding)
            # Create ID for this embedding
            row_id = row.get('respondent_id', row.get('id', f'row_{row_idx}'))
            embedding_id = f"{row_id}_{embed_col}"
            embedding_ids.append(embedding_id)
    
    if not embeddings:
        raise ValueError(f"No valid embeddings found in column {embed_col}")
    
    embeddings_array = np.array(embeddings)
    
    metadata = {
        "format": "single",
        "original_rows": len(df),
        "embedding_column": embed_col,
        "total_embeddings": len(embeddings),
        "embedding_ids": embedding_ids,  # IDs for mapping back
        "original_dataframe": df  # Keep reference for mapping
    }
    
    return embeddings_array, metadata


def _parse_embedding(value: Any) -> np.ndarray:
    """Parse an embedding value from various formats.
    
    Args:
        value: Embedding value (string, list, array, etc.)
        
    Returns:
        Numpy array or None if parsing failed
    """
    # Handle None/NaN values
    if value is None or (hasattr(value, '__len__') and len(value) == 0):
        return None
        
    # Handle pandas NA
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        # pd.isna can fail on some array types
        pass
    
    try:
        if isinstance(value, str):
            # Try to parse as literal list
            parsed = ast.literal_eval(value)
            return np.array(parsed, dtype=float)
        
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            return np.array(value, dtype=float)
        
        elif isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            return value.astype(float)
        
        else:
            # Try direct conversion
            arr = np.array(value, dtype=float)
            if arr.size == 0:
                return None
            return arr
    
    except Exception as e:
        print(f"Warning: Failed to parse embedding: {e}")
        return None