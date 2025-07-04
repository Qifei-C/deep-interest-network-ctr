"""
Attention Mechanism Implementation for Deep Interest Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class AttentionLayer(nn.Module):
    """
    Attention layer for Deep Interest Network
    """
    
    def __init__(self, embedding_dim: int, hidden_dims: List[int] = [80, 40], dropout: float = 0.1):
        """
        Initialize attention layer
        
        Args:
            embedding_dim: Dimension of embeddings
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super(AttentionLayer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        
        # Build attention network
        layers = []
        input_dim = embedding_dim * 4  # [user_embedding, item_embedding, user_item_concat, user_item_element_wise]
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.attention_net = nn.Sequential(*layers)
        
    def forward(self, user_embedding: torch.Tensor, item_embeddings: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of attention mechanism
        
        Args:
            user_embedding: User embedding [batch_size, embedding_dim]
            item_embeddings: Item embeddings [batch_size, seq_len, embedding_dim]
            mask: Mask for padding [batch_size, seq_len]
            
        Returns:
            Tuple of (attended_embedding, attention_weights)
        """
        batch_size, seq_len, embedding_dim = item_embeddings.shape
        
        # Expand user embedding to match sequence length
        user_expanded = user_embedding.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, embedding_dim]
        
        # Create attention input features
        # 1. User embedding
        user_features = user_expanded
        
        # 2. Item embedding
        item_features = item_embeddings
        
        # 3. Concatenation of user and item
        concat_features = torch.cat([user_expanded, item_embeddings], dim=-1)
        
        # 4. Element-wise product
        element_wise_features = user_expanded * item_embeddings
        
        # Combine all features
        attention_input = torch.cat([
            user_features, item_features, concat_features, element_wise_features
        ], dim=-1)  # [batch_size, seq_len, embedding_dim * 4]
        
        # Compute attention scores
        attention_scores = self.attention_net(attention_input).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len]
        
        # Apply attention to get weighted sum
        attended_embedding = torch.sum(
            attention_weights.unsqueeze(-1) * item_embeddings, dim=1
        )  # [batch_size, embedding_dim]
        
        return attended_embedding, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize multi-head attention
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        
        assert embedding_dim % num_heads == 0
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.query_linear = nn.Linear(embedding_dim, embedding_dim)
        self.key_linear = nn.Linear(embedding_dim, embedding_dim)
        self.value_linear = nn.Linear(embedding_dim, embedding_dim)
        self.output_linear = nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention
        
        Args:
            query: Query tensor [batch_size, seq_len_q, embedding_dim]
            key: Key tensor [batch_size, seq_len_k, embedding_dim]
            value: Value tensor [batch_size, seq_len_v, embedding_dim]
            mask: Attention mask [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Linear transformations and reshape for multi-head
        Q = self.query_linear(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_linear(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_linear(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        output, attention_weights = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embedding_dim
        )
        
        # Final linear transformation
        output = self.output_linear(output)
        
        return output, attention_weights
    
    def _scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled dot-product attention
        
        Args:
            Q: Query [batch_size, num_heads, seq_len_q, head_dim]
            K: Key [batch_size, num_heads, seq_len_k, head_dim]
            V: Value [batch_size, num_heads, seq_len_v, head_dim]
            mask: Attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for sequence modeling
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize self-attention
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(SelfAttention, self).__init__()
        
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of self-attention
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
            mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, embedding_dim]
        """
        # Self-attention
        attended, _ = self.multi_head_attention(x, x, x, mask)
        
        # Add & Norm
        output = self.layer_norm(x + self.dropout(attended))
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence modeling
    """
    
    def __init__(self, embedding_dim: int, max_length: int = 1000):
        """
        Initialize positional encoding
        
        Args:
            embedding_dim: Dimension of embeddings
            max_length: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           (-np.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor [seq_len, batch_size, embedding_dim]
            
        Returns:
            Output with positional encoding
        """
        return x + self.pe[:x.size(0), :]


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward network
    """
    
    def __init__(self, embedding_dim: int, num_heads: int = 8, ff_dim: int = 512, dropout: float = 0.1):
        """
        Initialize transformer block
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            ff_dim: Dimension of feed-forward network
            dropout: Dropout rate
        """
        super(TransformerBlock, self).__init__()
        
        self.self_attention = SelfAttention(embedding_dim, num_heads, dropout)
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim)
        )
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of transformer block
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
            mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, embedding_dim]
        """
        # Self-attention
        attended = self.self_attention(x, mask)
        
        # Feed-forward network
        ff_output = self.ff_network(attended)
        
        # Add & Norm
        output = self.layer_norm(attended + self.dropout(ff_output))
        
        return output


class DINAttentionPooling(nn.Module):
    """
    Deep Interest Network style attention pooling
    """
    
    def __init__(self, embedding_dim: int, hidden_dims: List[int] = [200, 80], activation: str = 'relu'):
        """
        Initialize DIN attention pooling
        
        Args:
            embedding_dim: Dimension of embeddings
            hidden_dims: Hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(DINAttentionPooling, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build attention network
        layers = []
        input_dim = embedding_dim * 3  # [item, target, item-target interaction]
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.attention_layers = nn.ModuleList()
        input_dim = embedding_dim * 3
        for hidden_dim in hidden_dims:
            self.attention_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.attention_layers.append(nn.Linear(input_dim, 1))
        
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(dim) for dim in hidden_dims])
        
    def forward(self, item_embeddings: torch.Tensor, target_embedding: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of attention pooling
        
        Args:
            item_embeddings: Item embeddings [batch_size, seq_len, embedding_dim]
            target_embedding: Target item embedding [batch_size, embedding_dim]
            mask: Mask for padding [batch_size, seq_len]
            
        Returns:
            Tuple of (pooled_embedding, attention_weights)
        """
        batch_size, seq_len, embedding_dim = item_embeddings.shape
        
        # Expand target embedding
        target_expanded = target_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Create interaction features
        interaction = item_embeddings * target_expanded
        
        # Concatenate features
        attention_input = torch.cat([
            item_embeddings, target_expanded, interaction
        ], dim=-1)  # [batch_size, seq_len, embedding_dim * 3]
        
        # Reshape for batch norm
        attention_input = attention_input.view(-1, self.embedding_dim * 3)
        
        # Forward through attention network
        x = attention_input
        for i, (layer, bn) in enumerate(zip(self.attention_layers[:-1], self.batch_norms)):
            x = layer(x)
            x = self.activation(x)
            x = bn(x)
        
        # Final layer
        attention_scores = self.attention_layers[-1](x)  # [batch_size * seq_len, 1]
        attention_scores = attention_scores.view(batch_size, seq_len)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to get weighted sum
        pooled_embedding = torch.sum(
            attention_weights.unsqueeze(-1) * item_embeddings, dim=1
        )  # [batch_size, embedding_dim]
        
        return pooled_embedding, attention_weights