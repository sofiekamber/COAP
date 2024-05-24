#!/bin/bash

set -e
set -o xtrace


# Run your experiment
# Define variables
LOG_FILE="./TRAINED_MODELS/hand_models/logfile_left.log"
CONFIG_FILE="./configs/hand_models/mano_left.yml"
#OUT_DIR="./TRAINED_MODELS/hand_models/mano/left_wo_transl"
#LOG_FILE="./TRAINED_MODELS/hand_models/logfile_right.log"
#CONFIG_FILE="./configs/hand_models/mano_right.yml"
OUT_DIR="./TRAINED_MODELS/hand_models/mano/left_visualize"
MODELS="/home/sofie-kamber/Projects/datasets/team_3/models"
DATA_ROOT_smplx="/home/sofie-kamber/Projects/datasets/team_3/data_mano"

# Run the Python command
python train.py "$CONFIG_FILE" --out_dir "$OUT_DIR" --model_path "${MODELS}" --data_root "${DATA_ROOT_smplx}" --run_eval --eval_export_visuals --accelerator 'gpu' --max_epochs 30 --devices 1 2>&1 | tee -a "${LOG_FILE}"

echo "Done."
echo FINISHED at $(date)