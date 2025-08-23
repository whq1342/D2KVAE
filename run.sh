#!/bin/bash
LOG_DIR="./logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

python main.py --number 1 --label_rate 0.2 --seq_len 30 --stride 5 --label_weight 5 --kl_x_weight 0.1 --kl_y_weight 0.001 --koopman_weight 0.01 --z_x_dim 8 --z_y_dim 9 --dense_x_g 12 --dense_h_x 16 --dense_z_y 15 --dense_h_y 16 --gpu 0 > "${LOG_DIR}/${TIMESTAMP}_exp_1_label_rate_0.2.log" 2>&1 &
python main.py --number 2 --label_rate 0.3 --seq_len 30 --stride 5 --label_weight 5 --kl_x_weight 0.1 --kl_y_weight 0.001 --koopman_weight 0.01 --z_x_dim 7 --z_y_dim 13 --dense_x_g 12 --dense_h_x 32 --dense_z_y 25 --dense_h_y 16 --gpu 1 > "${LOG_DIR}/${TIMESTAMP}_exp_2_label_rate_0.3.log" 2>&1 &
python main.py --number 3 --label_rate 0.4 --seq_len 30 --stride 5 --label_weight 5 --kl_x_weight 0.001 --kl_y_weight 0.001 --koopman_weight 0.001 --z_x_dim 9 --z_y_dim 15 --dense_x_g 12 --dense_h_x 16 --dense_z_y 20 --dense_h_y 32 --gpu 2 > "${LOG_DIR}/${TIMESTAMP}_exp_3_label_rate_0.4.log" 2>&1 &
python main.py --number 4 --label_rate 0.5 --seq_len 30 --stride 5 --label_weight 2 --kl_x_weight 0.001 --kl_y_weight 1 --koopman_weight 0.01 --z_x_dim 12 --z_y_dim 20 --dense_x_g 12 --dense_h_x 8 --dense_z_y 25 --dense_h_y 64 --gpu 3 > "${LOG_DIR}/${TIMESTAMP}_exp_4_label_rate_0.5.log" 2>&1
