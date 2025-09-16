LOG_DIR="./logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

python main.py --number 1 --seed 0 --label_rate 0.2 --gpu 0 > "${LOG_DIR}/${TIMESTAMP}_number_1_seed_0_label_rate_0.2.log" 2>&1 &
python main.py --number 2 --seed 0 --label_rate 0.3 --gpu 1 > "${LOG_DIR}/${TIMESTAMP}_number_2_seed_0_label_rate_0.3.log" 2>&1 &
python main.py --number 3 --seed 0 --label_rate 0.4 --gpu 2 > "${LOG_DIR}/${TIMESTAMP}_number_3_seed_0_label_rate_0.4.log" 2>&1 &
python main.py --number 4 --seed 0 --label_rate 0.5 --gpu 3 > "${LOG_DIR}/${TIMESTAMP}_number_4_seed_0_label_rate_0.5.log" 2>&1 &
python main.py --number 5 --seed 1 --label_rate 0.2 --gpu 4 > "${LOG_DIR}/${TIMESTAMP}_number_5_seed_1_label_rate_0.2.log" 2>&1 &
python main.py --number 6 --seed 1 --label_rate 0.3 --gpu 5 > "${LOG_DIR}/${TIMESTAMP}_number_6_seed_1_label_rate_0.3.log" 2>&1 &
python main.py --number 7 --seed 1 --label_rate 0.4 --gpu 6 > "${LOG_DIR}/${TIMESTAMP}_number_7_seed_1_label_rate_0.4.log" 2>&1 &
python main.py --number 8 --seed 1 --label_rate 0.5 --gpu 7 > "${LOG_DIR}/${TIMESTAMP}_number_8_seed_1_label_rate_0.5.log" 2>&1 &
python main.py --number 9 --seed 2 --label_rate 0.2 --gpu 0 > "${LOG_DIR}/${TIMESTAMP}_number_9_seed_2_label_rate_0.2.log" 2>&1 &
python main.py --number 10 --seed 2 --label_rate 0.3 --gpu 1 > "${LOG_DIR}/${TIMESTAMP}_number_10_seed_2_label_rate_0.3.log" 2>&1 &
python main.py --number 11 --seed 2 --label_rate 0.4 --gpu 2 > "${LOG_DIR}/${TIMESTAMP}_number_11_seed_2_label_rate_0.4.log" 2>&1 &
python main.py --number 12 --seed 2 --label_rate 0.5 --gpu 3 > "${LOG_DIR}/${TIMESTAMP}_number_12_seed_2_label_rate_0.5.log" 2>&1 &