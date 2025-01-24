# SYNTHETIC Dataset
python \
  -u run.py \
  --task_name addition \
  --is_training 1 \
  --root_path ./dataset/SYNTHETIC/ \
  --model_id SYNTHETIC \
  --model Medformer \
  --data SYNTHETIC \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --patch_len_list 4,8,16 \
  --augmentations flip,shuffle,jitter,drop \
  --swa \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

# UCI-HAR Dataset
python \
  -u run.py \
  --task_name addition \
  --is_training 1 \
  --root_path ./dataset/UCI-HAR/ \
  --model_id UCI-HAR \
  --model Medformer \
  --data UCI-HAR \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --patch_len_list 4,4,8,8 \
  --augmentations flip,shuffle,jitter,drop \
  --swa \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10


# FLAAP Dataset
python \
  -u run.py \
  --task_name addition \
  --is_training 1 \
  --root_path ./dataset/FLAAP/ \
  --model_id FLAAP \
  --model Medformer \
  --data FLAAP \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --patch_len_list 2,4,8 \
  --augmentations mask,drop \
  --swa \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10