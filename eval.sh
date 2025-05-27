
# CUDA_VISIBLE_DEVICES=6 python eval_action_predictability.py \
#        --ckpt    output/PegInsertionSide-v2/Offline-DiffusionPolicy-UNet/250429-162348_1_test/checkpoints/best_eval_success_rate.pt \
#        --demo-h5 data/PegInsertionSide/trajectory.h5

       

CUDA_VISIBLE_DEVICES=6 python eval_action_predictability.py \
       --ckpt    output/PegInsertionSide-v2/Offline-DiffusionPolicy-UNet/250522-220135_1_ptp/checkpoints/best_eval_success_rate.pt \
       --demo-h5 data/PegInsertionSide/trajectory.h5
