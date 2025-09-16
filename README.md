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
- **Dual latent space architecture for independent representation**
- **Differentiated priors within a dynamic VAE framework**
- **Enhanced Koopman constraint with random steps**
- **Demonstrated effectiveness on industrial benchmarks**

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



## 📈 Results

The model achieves excellent performance across different label rates:

| Label Rate | Test RMSE | Test MAE | R² Score |
|------------|-----------|----------|----------|
| 50%        | 2.320±0.092   | 1.750±0.072  | 82.468±0.311   |
| 40%        | 2.357±0.086   | 1.763±0.072  | 80.356±1.971   |
| 30%        | 2.407±0.037   | 1.769±0.035  | 78.227±1.289   |
| 20%        | 2.458±0.134   | 1.858±0.108  | 77.237±2.043   |

## 📋 Output Logs
Training progress and results are saved in the `./logs/` directory with timestamped filenames:

20250916_190905_number_1_seed_0_label_rate_0.2.log
20250916_190905_number_2_seed_0_label_rate_0.3.log
20250916_190905_number_3_seed_0_label_rate_0.4.log
20250916_190905_number_4_seed_0_label_rate_0.5.log
20250916_190905_number_5_seed_1_label_rate_0.2.log
20250916_190905_number_6_seed_1_label_rate_0.3.log
20250916_190905_number_7_seed_1_label_rate_0.4.log
20250916_190905_number_8_seed_1_label_rate_0.5.log
20250916_190905_number_9_seed_2_label_rate_0.2.log
20250916_190905_number_10_seed_2_label_rate_0.3.log
20250916_190905_number_11_seed_2_label_rate_0.4.log
20250916_190905_number_12_seed_2_label_rate_0.5.log

Each log contains:

- Training and validation losses (total, reconstruction, KL divergences, label, Koopman)  
- Learning rate scheduling information  
- Final evaluation metrics (RMSE, MAE, R²)  
- Training and testing times  


## 📝 Citation
If you use this code in your research, please cite:

```
```
