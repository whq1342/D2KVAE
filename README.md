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
- **Dual latent space architecture for independent representation**: We construct a dual latent structure that explicitly separates process-related and quality-related representations, allowing more accurate and independent characterization of key quality information.
- **Differentiated priors within a dynamic VAE framework**: Distinct prior distributions are designed for the two latent spaces within dynamic VAE framework, which enhances the ability to capture uniform and transferable dynamic features.
- **Enhanced Koopman constraint with random steps**: An Koopman-based regularization is introduced for random-step dynamics, enforcing implicit local linear continuity of quality-related states without relying on pseudo-labels and thus avoiding explicit inductive bias.
- **Demonstrated effectiveness on industrial benchmarks**: The experiments on real-world industrial process datasets demonstrate that the proposed method not only achieves optimal prediction accuracy and stability, but also exhibits strong extrapolation capabilities.

## 🛠️ Installation
```
pip install torch==2.2.2 numpy==1.26.4 matplotlib scikit-learn
```

## 📊 Usage

### Quick Start  
Run the batch experiments with different label rates:
```
# Make the script executable  
chmod +x run.sh  

# Execute all experiments  
./run.sh  
```

The experiment was conducted on:

- **GPU**: NVIDIA 3090  
- **CPU**: Intel(R) Xeon(R) Silver 4214 CPU @ 2.70 GHz  
- **OS**: Ubuntu 18.04.5 LTS  
- **Python**: 3.10.8  



## ⚙️ Configuration Parameters

| Parameter        | Description                        | Default                    |
|------------------|------------------------------------|----------------------------|
| --label_rate     | Percentage of labeled data         | 0.2, 0.3, 0.4, 0.5         |
| --seq_len        | Sequence length for time series    | 20, 30, 40                 |
| --stride         | Sliding window stride              | 1, 5                       |
| --label_weight   | Weight for supervised loss         | 1, 2, 5                    |
| --kl_x_weight    | Weight for x-space KL divergence   | 1, 0.1, 0.01, 0.001, 0.0001|
| --kl_y_weight    | Weight for y-space KL divergence   | 1, 0.1, 0.01, 0.001, 0.0001|
| --koopman_weight | Weight for Koopman loss            | 1, 0.1, 0.01, 0.001, 0.0001|
| --z_x_dim        | Latent dimension for x             | 6-15                       |
| --z_y_dim        | Latent dimension for y             | 7-20                       |

## 📈 Results

The model achieves excellent performance across different label rates:

| Label Rate | Test RMSE | Test MAE | R² Score | Training Time |
|------------|-----------|----------|----------|---------------|
| 20%        | 2.33184   | 1.76085  | 81.19%   | 324.81s       |
| 30%        | 2.43632   | 1.73426  | 79.62%   | 309.54s       |
| 40%        | 2.25353   | 1.67239  | 81.85%   | 319.52s       |
| 50%        | 2.37242   | 1.82313  | 82.26%   | 338.34s       |

## 📋 Output Logs
Training progress and results are saved in the `./logs/` directory with timestamped filenames:

20250823_214416_exp_1_label_rate_0.2.log  
20250823_214416_exp_2_label_rate_0.3.log  
20250823_214416_exp_3_label_rate_0.4.log  
20250823_214416_exp_4_label_rate_0.5.log  

Each log contains:

- Training and validation losses (total, reconstruction, KL divergences, label, Koopman)  
- Learning rate scheduling information  
- Final evaluation metrics (RMSE, MAE, R²)  
- Training and testing times  


## 📝 Citation
If you use this code in your research, please cite:

```
```
