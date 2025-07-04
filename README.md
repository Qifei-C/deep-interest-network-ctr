# Deep Interest Network (DIN) for CTR Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-1.5+-purple.svg)](https://www.pytorchlightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of Deep Interest Network (DIN) for Click-Through Rate (CTR) prediction. DIN uses attention mechanisms to capture user interests dynamically based on candidate advertisements, achieving state-of-the-art performance in industrial CTR prediction scenarios.

## 🎯 Features

- **Attention Mechanism**: Dynamic interest modeling with local activation
- **Industrial Scale**: Optimized for large-scale CTR prediction
- **Feature Engineering**: Handles categorical and numerical features
- **PyTorch Lightning**: Professional training framework
- **Production Ready**: Efficient inference and deployment

## 🚀 Quick Start

```python
from src.din import DIN, CTRDataset

# Initialize DIN model
model = DIN(
    categorical_features=['user_id', 'item_id', 'category_id'],
    numerical_features=['price', 'rating'],
    embedding_dim=64,
    hidden_dims=[512, 256, 128]
)

# Prepare CTR dataset
dataset = CTRDataset(
    user_behavior_data=behavior_data,
    ad_data=ad_data,
    labels=click_labels
)

# Train model
trainer = pl.Trainer(gpus=1, max_epochs=50)
trainer.fit(model, dataset)

# Predict CTR
ctr_scores = model.predict(user_features, ad_features)
```

## 🏗 Architecture

### DIN Components

1. **Embedding Layer**
   - Sparse feature embeddings
   - Dense feature normalization
   - Feature interaction modeling

2. **Attention Network**
   - Local activation unit
   - Attention weights for historical behaviors
   - Dynamic interest representation

3. **Deep Neural Network**
   - Multi-layer perceptron
   - Batch normalization and dropout
   - Output layer for CTR prediction

4. **Loss Function**
   - Binary cross-entropy for click prediction
   - Regularization for embedding stability

## 📊 Performance Benchmarks

Performance metrics will vary based on your specific dataset, feature engineering, and model configuration. The DIN architecture is designed to achieve competitive results on CTR prediction tasks.

## 📁 Project Structure

```
deep-interest-network-ctr/
├── src/                    # Source code
│   ├── din.py             # Main DIN model
│   ├── attention.py       # Attention mechanisms
│   └── dataset.py         # Data processing
├── examples/              # Example scripts
├── tests/                 # Test suite
├── docs/                  # Documentation
├── data/                  # Data directory
├── models/                # Trained models
└── README.md             # This file
```

## 🔧 Advanced Features

### Custom Attention Units
```python
# Configure attention mechanism
din_model = DIN(
    attention_type='local_activation',
    attention_hidden_dims=[128, 64],
    use_dice_activation=True
)
```

### Multi-Task Learning
```python
# Joint CTR and CVR prediction
model = MultiTaskDIN(
    tasks=['ctr', 'cvr'],
    shared_bottom_dims=[512, 256],
    task_specific_dims=[128, 64]
)
```

### Industrial Deployment
```python
# Export for serving
model.export_onnx('models/din_production.onnx')

# Load for inference
predictor = DINPredictor('models/din_production.onnx')
ctr_score = predictor.predict(user_features, ad_features)
```

## 🏭 Industrial Applications

- **Online Advertising**: Real-time ad CTR prediction
- **Recommendation Systems**: Click probability estimation
- **E-commerce**: Product recommendation ranking
- **Content Platforms**: Content engagement prediction

## 📈 Research Applications

This implementation supports research in:
- Attention mechanisms for CTR prediction
- User behavior modeling in advertising
- Large-scale machine learning systems
- Multi-task learning in recommendations

## 🤝 Contributing

We welcome contributions to improve DIN implementation! Please submit issues and pull requests.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📚 Citation

```bibtex
@inproceedings{zhou2018deep,
  title={Deep interest network for click-through rate prediction},
  author={Zhou, Guorui and Zhu, Xiaoqiang and Song, Chenru and Fan, Ying and Zhu, Han and Ma, Xiao and Yan, Yanghui and Jin, Jianqiang and Li, Han and Gai, Kun},
  booktitle={KDD},
  year={2018}
}
```

---

📊 **Click-Through Rate Prediction with Attention**