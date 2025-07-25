"""
Data Loader Module for SafeData Pipeline
Handles loading and validation of various data formats
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import openpyxl
import sqlite3
from io import StringIO
import chardet

class DataLoader:
    """Data loading and validation class"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.tsv', '.txt', '.jsonl', '.feather', '.pkl', '.pickle']
        self.current_dataset = None
        
    def load_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from various file formats
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Dictionary with success status and data information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        self.logger.info(f"Loading data from {file_path}")
        
        try:
            if file_extension in ['.csv', '.txt']:
                dataset = self._load_csv(file_path)
            elif file_extension == '.tsv':
                dataset = self._load_tsv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                dataset = self._load_excel(file_path)
            elif file_extension == '.json':
                dataset = self._load_json(file_path)
            elif file_extension == '.jsonl':
                dataset = self._load_jsonl(file_path)
            elif file_extension == '.parquet':
                dataset = self._load_parquet(file_path)
            elif file_extension == '.feather':
                dataset = self._load_feather(file_path)
            elif file_extension in ['.pkl', '.pickle']:
                dataset = self._load_pickle(file_path)
            else:
                raise ValueError(f"Handler not implemented for {file_extension}")
            
            # Validate and clean the dataset
            dataset = self._validate_and_clean(dataset)
            
            # Store current dataset
            self.current_dataset = dataset
            
            # Create data info for web response
            data_info = {
                'filename': file_path.name,
                'rows': int(len(dataset)),
                'columns': int(len(dataset.columns)),
                'size': int(file_path.stat().st_size),
                'format': file_extension.upper().lstrip('.'),
                'column_names': list(dataset.columns),
                'data_types': {str(k): str(v) for k, v in dataset.dtypes.to_dict().items()},
                'memory_usage': int(dataset.memory_usage(deep=True).sum())
            }
            
            self.logger.info(f"Successfully loaded {len(dataset)} records with {len(dataset.columns)} columns")
            
            return {
                'success': True,
                'data_info': data_info,
                'message': f'Successfully loaded {len(dataset)} records'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'message': f'Failed to load data: {str(e)}'
            }
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'
    
    def _load_csv(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file with automatic delimiter detection"""
        encoding = self._detect_encoding(file_path)
        
        # Try different delimiters
        delimiters = [',', ';', '\t', '|']
        
        for delimiter in delimiters:
            try:
                df = pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    encoding=encoding,
                    low_memory=False,
                    na_values=['', 'NULL', 'null', 'NaN', 'nan', 'N/A', 'n/a']
                )
                
                # Check if we got reasonable column separation
                if len(df.columns) > 1:
                    return df
                    
            except Exception as e:
                self.logger.debug(f"Failed to load with delimiter '{delimiter}': {str(e)}")
                continue
        
        # Fallback to default comma delimiter
        return pd.read_csv(
            file_path,
            encoding=encoding,
            low_memory=False,
            na_values=['', 'NULL', 'null', 'NaN', 'nan', 'N/A', 'n/a']
        )
    
    def _load_tsv(self, file_path: Path) -> pd.DataFrame:
        """Load TSV (Tab-Separated Values) file"""
        encoding = self._detect_encoding(file_path)
        
        return pd.read_csv(
            file_path,
            delimiter='\t',
            encoding=encoding,
            low_memory=False,
            na_values=['', 'NULL', 'null', 'NaN', 'nan', 'N/A', 'n/a']
        )
    
    def _load_excel(self, file_path: Path) -> pd.DataFrame:
        """Load Excel file (xlsx/xls)"""
        try:
            # Try to read all sheets and use the first non-empty one
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(
                        file_path,
                        sheet_name=sheet_name,
                        na_values=['', 'NULL', 'null', 'NaN', 'nan', 'N/A', 'n/a']
                    )
                    
                    if not df.empty and len(df.columns) > 0:
                        self.logger.info(f"Loaded sheet: {sheet_name}")
                        return df
                        
                except Exception as e:
                    self.logger.debug(f"Failed to load sheet {sheet_name}: {str(e)}")
                    continue
            
            # If no sheet worked, try default
            return pd.read_excel(
                file_path,
                na_values=['', 'NULL', 'null', 'NaN', 'nan', 'N/A', 'n/a']
            )
            
        except Exception as e:
            raise ValueError(f"Failed to load Excel file: {str(e)}")
    
    def _load_json(self, file_path: Path) -> pd.DataFrame:
        """Load JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of records
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Check if it's a dictionary with data key
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                elif 'records' in data:
                    df = pd.DataFrame(data['records'])
                else:
                    # Try to normalize the nested structure
                    try:
                        df = pd.json_normalize(data)
                    except:
                        # Treat as single record
                        df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported JSON structure")
            
            return df
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to load JSON file: {str(e)}")
    
    def _load_jsonl(self, file_path: Path) -> pd.DataFrame:
        """Load JSON Lines file"""
        try:
            records = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line.strip()))
            return pd.DataFrame(records)
        except Exception as e:
            raise ValueError(f"Failed to load JSONL file: {str(e)}")
    
    def _load_parquet(self, file_path: Path) -> pd.DataFrame:
        """Load Parquet file"""
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load Parquet file: {str(e)}")
    
    def _load_feather(self, file_path: Path) -> pd.DataFrame:
        """Load Feather file"""
        try:
            return pd.read_feather(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load Feather file: {str(e)}")
    
    def _load_pickle(self, file_path: Path) -> pd.DataFrame:
        """Load Pickle file"""
        try:
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, dict):
                return pd.DataFrame(data)
            elif isinstance(data, list):
                return pd.DataFrame(data)
            else:
                raise ValueError("Unsupported pickle data type")
        except Exception as e:
            raise ValueError(f"Failed to load Pickle file: {str(e)}")
    
    def _validate_and_clean(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the loaded dataset"""
        
        if dataset.empty:
            raise ValueError("Dataset is empty")
        
        original_shape = dataset.shape
        
        # Remove completely empty rows and columns
        dataset = dataset.dropna(how='all')  # Remove rows with all NaN
        dataset = dataset.dropna(axis=1, how='all')  # Remove columns with all NaN
        
        # Clean column names
        dataset.columns = [self._clean_column_name(col) for col in dataset.columns]
        
        # Remove duplicate column names
        dataset = self._handle_duplicate_columns(dataset)
        
        # Basic data type inference and conversion
        dataset = self._infer_data_types(dataset)
        
        # Log cleaning summary
        if dataset.shape != original_shape:
            self.logger.info(f"Dataset cleaned: {original_shape} -> {dataset.shape}")
        
        return dataset
    
    def _clean_column_name(self, column_name: str) -> str:
        """Clean column names"""
        if pd.isna(column_name):
            return "unnamed_column"
        
        # Convert to string and clean
        column_name = str(column_name).strip()
        
        # Replace problematic characters
        column_name = column_name.replace(' ', '_')
        column_name = column_name.replace('-', '_')
        column_name = column_name.replace('.', '_')
        column_name = ''.join(c for c in column_name if c.isalnum() or c == '_')
        
        # Ensure it doesn't start with a number
        if column_name and column_name[0].isdigit():
            column_name = 'col_' + column_name
        
        # Handle empty names
        if not column_name:
            column_name = "unnamed_column"
        
        return column_name
    
    def _handle_duplicate_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate column names"""
        columns = list(dataset.columns)
        seen = set()
        new_columns = []
        
        for col in columns:
            original_col = col
            counter = 1
            
            while col in seen:
                col = f"{original_col}_{counter}"
                counter += 1
            
            seen.add(col)
            new_columns.append(col)
        
        dataset.columns = new_columns
        return dataset
    
    def _infer_data_types(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Infer and convert appropriate data types"""
        
        for column in dataset.columns:
            try:
                # Skip if already numeric
                if pd.api.types.is_numeric_dtype(dataset[column]):
                    continue
                
                # Try to convert to numeric
                if self._is_numeric_column(dataset[column]):
                    dataset[column] = pd.to_numeric(dataset[column], errors='coerce')
                    continue
                
                # Try to convert to datetime
                if self._is_datetime_column(dataset[column]):
                    dataset[column] = pd.to_datetime(dataset[column], errors='coerce')
                    continue
                
                # Convert to categorical if low cardinality
                if self._is_categorical_column(dataset[column]):
                    dataset[column] = dataset[column].astype('category')
                
            except Exception as e:
                self.logger.debug(f"Type inference failed for column {column}: {str(e)}")
                continue
        
        return dataset
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if a column should be treated as numeric"""
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return False
        
        # Try to convert a sample to numeric
        sample_size = min(100, len(non_null))
        sample = non_null.head(sample_size)
        
        try:
            pd.to_numeric(sample, errors='raise')
            return True
        except:
            return False
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Check if a column should be treated as datetime"""
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return False
        
        # Check for common datetime patterns
        sample_size = min(50, len(non_null))
        sample = non_null.head(sample_size)
        
        try:
            parsed = pd.to_datetime(sample, errors='coerce')
            success_rate = parsed.notna().sum() / len(parsed)
            return success_rate > 0.8
        except:
            return False
    
    def _is_categorical_column(self, series: pd.Series) -> bool:
        """Check if a column should be treated as categorical"""
        non_null = series.dropna()
        
        if len(non_null) == 0:
            return False
        
        # Convert to categorical if cardinality is low relative to data size
        unique_ratio = series.nunique() / len(series)
        return unique_ratio < 0.5 and series.nunique() < 100
    
    def get_dataset_info(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive dataset information"""
        
        info = {
            'shape': dataset.shape,
            'memory_usage_mb': dataset.memory_usage(deep=True).sum() / (1024 * 1024),
            'columns': {
                'total': len(dataset.columns),
                'numeric': len(dataset.select_dtypes(include=[np.number]).columns),
                'categorical': len(dataset.select_dtypes(include=['object', 'category']).columns),
                'datetime': len(dataset.select_dtypes(include=['datetime64']).columns)
            },
            'missing_values': {
                'total': dataset.isnull().sum().sum(),
                'percentage': (dataset.isnull().sum().sum() / (dataset.shape[0] * dataset.shape[1])) * 100
            },
            'duplicates': {
                'count': dataset.duplicated().sum(),
                'percentage': (dataset.duplicated().sum() / len(dataset)) * 100
            }
        }
        
        # Column-wise details
        column_details = {}
        for col in dataset.columns:
            column_details[col] = {
                'dtype': str(dataset[col].dtype),
                'non_null_count': dataset[col].count(),
                'null_count': dataset[col].isnull().sum(),
                'unique_count': dataset[col].nunique(),
                'unique_ratio': dataset[col].nunique() / len(dataset)
            }
            
            if pd.api.types.is_numeric_dtype(dataset[col]):
                column_details[col].update({
                    'min': dataset[col].min(),
                    'max': dataset[col].max(),
                    'mean': dataset[col].mean(),
                    'std': dataset[col].std()
                })
        
        info['column_details'] = column_details
        
        return info
    
    def save_dataset(self, dataset: pd.DataFrame, file_path: Union[str, Path], format: str = 'csv') -> None:
        """Save dataset to file"""
        
        file_path = Path(file_path)
        
        try:
            if format.lower() == 'csv':
                dataset.to_csv(file_path, index=False)
            elif format.lower() == 'excel':
                dataset.to_excel(file_path, index=False)
            elif format.lower() == 'json':
                dataset.to_json(file_path, orient='records', indent=2)
            elif format.lower() == 'parquet':
                dataset.to_parquet(file_path, index=False)
            else:
                raise ValueError(f"Unsupported save format: {format}")
            
            self.logger.info(f"Dataset saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save dataset: {str(e)}")
            raise
    
    def validate_quasi_identifiers(self, dataset: pd.DataFrame, quasi_identifiers: List[str]) -> Dict[str, Any]:
        """Validate quasi-identifier columns"""
        
        validation_results = {
            'valid_columns': [],
            'invalid_columns': [],
            'warnings': []
        }
        
        available_columns = list(dataset.columns)
        
        for qi in quasi_identifiers:
            if qi in available_columns:
                validation_results['valid_columns'].append(qi)
                
                # Check if column is suitable as QI
                unique_ratio = dataset[qi].nunique() / len(dataset)
                if unique_ratio > 0.9:
                    validation_results['warnings'].append(
                        f"Column '{qi}' has very high uniqueness ({unique_ratio:.2%}) - may not be effective for anonymization"
                    )
                elif unique_ratio < 0.01:
                    validation_results['warnings'].append(
                        f"Column '{qi}' has very low uniqueness ({unique_ratio:.2%}) - limited privacy protection"
                    )
            else:
                validation_results['invalid_columns'].append(qi)
        
        return validation_results
