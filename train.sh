CUDA_VISIBLE_DEVICES=7 python train_ptp.py --env-id PegInsertionSide-v2 --demo-path data/PegInsertionSide/trajectory.h5 --track  --past-horizon 8

# CUDA_VISIBLE_DEVICES=7 python train_ptp.py --env-id PegInsertionSide-v2 --demo-path data/PegInsertionSide/trajectory.h5 --track --past-horizon 0