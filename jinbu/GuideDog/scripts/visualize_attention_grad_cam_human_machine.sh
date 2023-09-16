# After launching the docker container 
EVAL_DIR='drama_output/transformer_baseline_pred_class_6layer_6head_lr0003/checkpoint-5-460/'
CHECKPOINT='drama_output/transformer_baseline_pred_class_6layer_6head_lr0003/checkpoint-5-460/model.bin'
CUDA_VISIBLE_DEVICES=5 python -m pdb src/tasks/visualize_attention_grad_cam_human_machine.py \
       --resume_checkpoint $CHECKPOINT  \
       --eval_model_dir $EVAL_DIR \
       --do_lower_case \
       --do_test 