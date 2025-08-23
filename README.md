# Dynamical Dual Latent Space Variational Autoencoder Model with Koopman Constraint for Semi-Supervised Soft Sensor Application
A PyTorch implementation of Dynamical Dual Latent Space Variational Autoencoder Model with Koopman Constraint for Semi-Supervised Soft Sensor Application.

## 📁 Project Structure
```text
├── data_pre/ # Data preprocessing modules
│ ├── data_pre.py # Data preprocessing script
│ └── dataset.py # Dataset class definition
├── dataset/ # Data directory
│ ├── SRU_data.npy # Process data in numpy format
│ └── SRU_data.txt # Process data in text format
├── figures/ # Generated figures and plots
├── inference/ # Inference and evaluation
│ └── quality_estimate.py # Future quality estimation
├── logs/ # Training logs and results
├── model/ # Model architecture
│ └── D2KVAE.py # Dynamical Dual Latent Space Variational Autoencoder with Koopman Constraint
├── models/ # Saved model checkpoints
├── train/ # Training utilities
│ └── trainer.py # Training loop and loss functions
├── utils/ # Utility functions
│ ├── FC.py # Fully connected layers
│ └── scheduler.py # Learning rate schedulers
├── main.py # Main execution script
└── run.sh # Batch experiment runner
```

## 🚀 Features

- **Semi-Supervised Learning**: Effective utilization of both labeled and unlabeled process data
- **Koopman Operator Theory**: Captures system dynamics in latent space
- **Variational Inference**: Robust uncertainty quantification
- **Multiple Label Rates**: Experiments with 20%, 30%, 40%, and 50% label availability
- **Comprehensive Metrics**: RMSE, MAE, R², training time, and computational complexity

## 🛠️ Installation
``
pip install torch numpy matplotlib scikit-learn
```
📊 Usage
Quick Start
Run the batch experiments with different label rates:
```bash
# Make the script executable
chmod +x run.sh
# Execute all experiments
./run.sh
```

⚙️ Configuration Parameters
Parameter	Description	Default
--label_rate	Percentage of labeled data	0.2, 0.3, 0.4, 0.5
--seq_len	Sequence length for time series	30
--stride	Sliding window stride	5
--label_weight	Weight for supervised loss	5
--kl_x_weight	Weight for x-space KL divergence	0.1
--kl_y_weight	Weight for y-space KL divergence	0.001
--koopman_weight	Weight for Koopman loss	0.01
--z_x_dim	Latent dimension for x	8-12
--z_y_dim	Latent dimension for y	9-20
📈 Results
The model achieves excellent performance across different label rates:

Label Rate	Test RMSE	Test MAE	R² Score	Training Time
20%	2.33184	1.76085	81.19%	324.81s
30%	2.43632	1.73426	79.62%	309.54s
40%	2.25353	1.67239	81.85%	319.52s
50%	2.37242	1.82313	82.26%	338.34s
📋 Output Logs
Training progress and results are saved in the ./logs/ directory with timestamped filenames:

20250823_214416_exp_1_label_rate_0.2.log

20250823_214416_exp_2_label_rate_0.3.log

20250823_214416_exp_3_label_rate_0.4.log

20250823_214416_exp_4_label_rate_0.5.log

Each log contains:

Training and validation losses (total, reconstruction, KL divergences, label, Koopman)

Learning rate scheduling information

Final evaluation metrics (RMSE, MAE, R²)

Training and testing times

🧠 Model Architecture
The model combines:

Variational Encoder: Maps input sequences to latent distributions

Koopman Operator: Learns linear dynamics in latent space

Decoder: Reconstructs inputs from latent representations

Regression Head: Predicts quality variables from latent states

🔧 Dependencies
Python 3.7+

PyTorch 2.2.2+

NumPy

Matplotlib

scikit-learn

📝 Citation
If you use this code in your research, please cite:

bibtex
@software{semi_supervised_soft_sensor,
  title = {Semi-Supervised Soft Sensor with Koopman VAE},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-username/repository-name}
}
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📧 Contact
For questions and support, please open an issue or contact your-email@example.com

text

This README provides a comprehensive overview of your project, including the structure, features, usage instructions, results, and technical details. The markdown format is ready to be used directly in your GitHub repository.
