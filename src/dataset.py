"""
Dataset handling for Deep Interest Network CTR prediction
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import pickle
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class CTRDataset(Dataset):
    """
    CTR prediction dataset for Deep Interest Network
    """
    
    def __init__(self, data: pd.DataFrame, user_col: str = 'user_id', 
                 item_col: str = 'item_id', label_col: str = 'label',
                 categorical_features: List[str] = None,
                 numerical_features: List[str] = None,
                 sequence_col: str = 'item_sequence',
                 max_sequence_length: int = 50):
        """
        Initialize CTR dataset
        
        Args:
            data: Input DataFrame
            user_col: User ID column name
            item_col: Item ID column name
            label_col: Label column name
            categorical_features: List of categorical feature columns
            numerical_features: List of numerical feature columns
            sequence_col: User sequence column name
            max_sequence_length: Maximum sequence length
        """
        self.data = data.copy()
        self.user_col = user_col
        self.item_col = item_col
        self.label_col = label_col
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
        self.sequence_col = sequence_col
        self.max_sequence_length = max_sequence_length
        
        # Encoders
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Processed data
        self.processed_data = None
        self.vocab_sizes = {}
        
        # Process the data
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess the dataset"""
        processed_data = self.data.copy()
        
        # Encode categorical features
        for feature in self.categorical_features + [self.user_col, self.item_col]:
            if feature in processed_data.columns:
                le = LabelEncoder()
                processed_data[feature] = le.fit_transform(processed_data[feature].astype(str))
                self.label_encoders[feature] = le
                self.vocab_sizes[feature] = len(le.classes_)
        
        # Scale numerical features
        if self.numerical_features:
            numerical_data = processed_data[self.numerical_features].fillna(0)
            processed_data[self.numerical_features] = self.scaler.fit_transform(numerical_data)
        
        # Process sequences
        if self.sequence_col in processed_data.columns:
            processed_data[self.sequence_col] = processed_data[self.sequence_col].apply(
                self._process_sequence
            )
        
        self.processed_data = processed_data
    
    def _process_sequence(self, sequence: Union[str, List]) -> List[int]:
        """
        Process user sequence
        
        Args:
            sequence: User interaction sequence
            
        Returns:
            Processed sequence as list of integers
        """
        if isinstance(sequence, str):
            # Parse string sequence (e.g., "1,2,3,4")
            if sequence.strip():
                items = [int(x.strip()) for x in sequence.split(',') if x.strip()]
            else:
                items = []
        elif isinstance(sequence, list):
            items = [int(x) for x in sequence if str(x).strip()]
        else:
            items = []
        
        # Encode items using item encoder
        if self.item_col in self.label_encoders:
            item_encoder = self.label_encoders[self.item_col]
            encoded_items = []
            for item in items:
                try:
                    encoded_items.append(item_encoder.transform([str(item)])[0])
                except ValueError:
                    # Unknown item, skip
                    continue
            items = encoded_items
        
        # Truncate or pad sequence
        if len(items) > self.max_sequence_length:
            items = items[-self.max_sequence_length:]
        
        return items
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing sample data
        """
        row = self.processed_data.iloc[idx]
        
        sample = {
            'user_id': torch.tensor(row[self.user_col], dtype=torch.long),
            'item_id': torch.tensor(row[self.item_col], dtype=torch.long),
            'label': torch.tensor(row[self.label_col], dtype=torch.float)
        }
        
        # Add categorical features
        for feature in self.categorical_features:
            if feature in row:
                sample[feature] = torch.tensor(row[feature], dtype=torch.long)
        
        # Add numerical features
        if self.numerical_features:
            numerical_values = [row[feature] for feature in self.numerical_features if feature in row]
            sample['numerical_features'] = torch.tensor(numerical_values, dtype=torch.float)
        
        # Add sequence
        if self.sequence_col in row:
            sequence = row[self.sequence_col]
            sequence_length = len(sequence)
            
            # Pad sequence
            padded_sequence = sequence + [0] * (self.max_sequence_length - len(sequence))
            sample['item_sequence'] = torch.tensor(padded_sequence, dtype=torch.long)
            sample['sequence_length'] = torch.tensor(sequence_length, dtype=torch.long)
            
            # Create mask
            mask = [1] * sequence_length + [0] * (self.max_sequence_length - sequence_length)
            sample['sequence_mask'] = torch.tensor(mask, dtype=torch.bool)
        
        return sample


class AmazonDataset(CTRDataset):
    """
    Amazon product recommendation dataset
    """
    
    def __init__(self, data_path: str, max_sequence_length: int = 50):
        """
        Initialize Amazon dataset
        
        Args:
            data_path: Path to Amazon dataset file
            max_sequence_length: Maximum sequence length
        """
        # Load data
        data = self._load_amazon_data(data_path)
        
        super().__init__(
            data=data,
            user_col='user_id',
            item_col='item_id',
            label_col='rating',
            categorical_features=['category'],
            sequence_col='item_sequence',
            max_sequence_length=max_sequence_length
        )
    
    def _load_amazon_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and preprocess Amazon dataset
        
        Args:
            data_path: Path to data file
            
        Returns:
            Processed DataFrame
        """
        # This is a placeholder implementation
        # In practice, you would load actual Amazon dataset
        
        if os.path.exists(data_path):
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                data = pd.read_json(data_path, lines=True)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        else:
            # Generate synthetic data for demonstration
            data = self._generate_synthetic_amazon_data()
        
        # Create user sequences
        data = self._create_user_sequences(data)
        
        return data
    
    def _generate_synthetic_amazon_data(self) -> pd.DataFrame:
        """Generate synthetic Amazon-like data"""
        np.random.seed(42)
        
        n_users = 1000
        n_items = 5000
        n_interactions = 50000
        
        data = {
            'user_id': np.random.randint(1, n_users + 1, n_interactions),
            'item_id': np.random.randint(1, n_items + 1, n_interactions),
            'rating': np.random.choice([1, 2, 3, 4, 5], n_interactions, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
            'timestamp': np.random.randint(1000000000, 1600000000, n_interactions),
            'category': np.random.choice(['Electronics', 'Books', 'Clothing', 'Home', 'Sports'], n_interactions)
        }
        
        df = pd.DataFrame(data)
        df = df.sort_values(['user_id', 'timestamp'])
        
        return df
    
    def _create_user_sequences(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create user interaction sequences
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with sequences
        """
        # Group by user and create sequences
        user_sequences = {}
        
        for user_id, group in data.groupby('user_id'):
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Create sequence for each interaction (excluding current item)
            sequences = []
            items = group['item_id'].tolist()
            
            for i in range(len(items)):
                # Sequence is all previous items
                sequence = items[:i] if i > 0 else []
                sequences.append(sequence)
            
            user_sequences[user_id] = sequences
        
        # Add sequences to data
        data = data.copy()
        data['item_sequence'] = data.apply(
            lambda row: user_sequences[row['user_id']][
                list(data[data['user_id'] == row['user_id']].index).index(row.name)
            ], axis=1
        )
        
        # Convert rating to binary label (4+ is positive)
        data['label'] = (data['rating'] >= 4).astype(int)
        
        return data


class MovielensDataset(CTRDataset):
    """
    MovieLens dataset for CTR prediction
    """
    
    def __init__(self, data_path: str, max_sequence_length: int = 50):
        """
        Initialize MovieLens dataset
        
        Args:
            data_path: Path to MovieLens dataset directory
            max_sequence_length: Maximum sequence length
        """
        # Load data
        data = self._load_movielens_data(data_path)
        
        super().__init__(
            data=data,
            user_col='user_id',
            item_col='movie_id',
            label_col='label',
            categorical_features=['gender', 'age_group', 'occupation', 'genre'],
            sequence_col='movie_sequence',
            max_sequence_length=max_sequence_length
        )
    
    def _load_movielens_data(self, data_path: str) -> pd.DataFrame:
        """Load MovieLens dataset"""
        try:
            # Try to load ratings
            ratings_path = os.path.join(data_path, 'ratings.dat') if os.path.isdir(data_path) else data_path
            
            if os.path.exists(ratings_path):
                ratings = pd.read_csv(ratings_path, sep='::', 
                                    names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                    engine='python')
            else:
                # Generate synthetic data
                ratings = self._generate_synthetic_movielens_data()
            
            # Load additional data if available
            users_path = os.path.join(os.path.dirname(ratings_path), 'users.dat')
            movies_path = os.path.join(os.path.dirname(ratings_path), 'movies.dat')
            
            if os.path.exists(users_path):
                users = pd.read_csv(users_path, sep='::', 
                                  names=['user_id', 'gender', 'age', 'occupation', 'zip'],
                                  engine='python')
                ratings = ratings.merge(users, on='user_id', how='left')
            
            if os.path.exists(movies_path):
                movies = pd.read_csv(movies_path, sep='::', 
                                   names=['movie_id', 'title', 'genres'],
                                   engine='python')
                movies['genre'] = movies['genres'].str.split('|').str[0]
                ratings = ratings.merge(movies[['movie_id', 'genre']], on='movie_id', how='left')
            
        except Exception:
            # Fallback to synthetic data
            ratings = self._generate_synthetic_movielens_data()
        
        # Create sequences and labels
        ratings = self._create_movie_sequences(ratings)
        
        return ratings
    
    def _generate_synthetic_movielens_data(self) -> pd.DataFrame:
        """Generate synthetic MovieLens-like data"""
        np.random.seed(42)
        
        n_users = 1000
        n_movies = 3000
        n_ratings = 100000
        
        data = {
            'user_id': np.random.randint(1, n_users + 1, n_ratings),
            'movie_id': np.random.randint(1, n_movies + 1, n_ratings),
            'rating': np.random.choice([1, 2, 3, 4, 5], n_ratings),
            'timestamp': np.random.randint(900000000, 1000000000, n_ratings),
            'gender': np.random.choice(['M', 'F'], n_ratings),
            'age': np.random.randint(18, 70, n_ratings),
            'occupation': np.random.randint(0, 21, n_ratings),
            'genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Horror', 'Romance'], n_ratings)
        }
        
        df = pd.DataFrame(data)
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], labels=['young', 'adult', 'middle', 'senior'])
        
        return df
    
    def _create_movie_sequences(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create movie sequences for users"""
        # Sort by user and timestamp
        data = data.sort_values(['user_id', 'timestamp'])
        
        # Create sequences
        user_sequences = {}
        
        for user_id, group in data.groupby('user_id'):
            movies = group['movie_id'].tolist()
            sequences = []
            
            for i in range(len(movies)):
                sequence = movies[:i] if i > 0 else []
                sequences.append(sequence)
            
            user_sequences[user_id] = sequences
        
        # Add sequences to data
        data['movie_sequence'] = data.apply(
            lambda row: user_sequences[row['user_id']][
                list(data[data['user_id'] == row['user_id']].index).index(row.name)
            ], axis=1
        )
        
        # Create binary labels
        data['label'] = (data['rating'] >= 4).astype(int)
        
        return data


class DataLoader:
    """
    Data loader for CTR datasets
    """
    
    def __init__(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0):
        """
        Initialize data loader
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
    
    def create_data_loader(self, dataset: CTRDataset) -> torch.utils.data.DataLoader:
        """
        Create PyTorch DataLoader
        
        Args:
            dataset: CTR dataset
            
        Returns:
            PyTorch DataLoader
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for batching
        
        Args:
            batch: List of samples
            
        Returns:
            Batched data
        """
        collated = {}
        
        # Get all keys from first sample
        keys = batch[0].keys()
        
        for key in keys:
            if key in batch[0]:
                values = [sample[key] for sample in batch]
                
                if values[0].dim() == 0:  # Scalar
                    collated[key] = torch.stack(values)
                else:  # Vector or matrix
                    collated[key] = torch.stack(values)
        
        return collated


class DataPreprocessor:
    """
    Data preprocessing utilities
    """
    
    @staticmethod
    def split_data(data: pd.DataFrame, train_ratio: float = 0.8, 
                   val_ratio: float = 0.1, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            data: Input DataFrame
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data
    
    @staticmethod
    def create_negative_samples(data: pd.DataFrame, user_col: str = 'user_id',
                              item_col: str = 'item_id', ratio: float = 1.0,
                              random_state: int = 42) -> pd.DataFrame:
        """
        Create negative samples for implicit feedback
        
        Args:
            data: Input DataFrame with positive samples
            user_col: User column name
            item_col: Item column name
            ratio: Negative to positive ratio
            random_state: Random state
            
        Returns:
            DataFrame with negative samples added
        """
        np.random.seed(random_state)
        
        # Get all users and items
        all_users = data[user_col].unique()
        all_items = data[item_col].unique()
        
        # Get positive interactions
        positive_pairs = set(zip(data[user_col], data[item_col]))
        
        # Generate negative samples
        negative_samples = []
        n_negatives = int(len(data) * ratio)
        
        while len(negative_samples) < n_negatives:
            user = np.random.choice(all_users)
            item = np.random.choice(all_items)
            
            if (user, item) not in positive_pairs:
                negative_samples.append({user_col: user, item_col: item, 'label': 0})
        
        # Create negative DataFrame
        negative_df = pd.DataFrame(negative_samples)
        
        # Add label to positive samples
        positive_df = data.copy()
        if 'label' not in positive_df.columns:
            positive_df['label'] = 1
        
        # Combine and shuffle
        combined_data = pd.concat([positive_df, negative_df], ignore_index=True)
        combined_data = combined_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        return combined_data
    
    @staticmethod
    def save_processed_data(data: pd.DataFrame, filepath: str):
        """Save processed data"""
        if filepath.endswith('.pkl'):
            data.to_pickle(filepath)
        elif filepath.endswith('.csv'):
            data.to_csv(filepath, index=False)
        else:
            raise ValueError("Unsupported file format. Use .pkl or .csv")
    
    @staticmethod
    def load_processed_data(filepath: str) -> pd.DataFrame:
        """Load processed data"""
        if filepath.endswith('.pkl'):
            return pd.read_pickle(filepath)
        elif filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        else:
            raise ValueError("Unsupported file format. Use .pkl or .csv")