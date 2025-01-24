 #APAVA Dataset
 python \
   -u run.py \
   --task_name classification \
   --root_path ./dataset/APAVA/ \
   --model_id APAVA-Indep \
   --model Medformer \
   --data APAVA \
   --e_layers 6 \
   --batch_size 32 \
   --d_model 128 \
   --d_ff 256 \
   --patch_len_list 2,2,2,4,4,4,16,16,16,16,32,32,32,32,32 \
   --augmentations none,drop0.35 \
   --swa \
   --des 'Exp' \
   --itr 5 \
   --learning_rate 0.0001 \
   --train_epochs 100 \
   --patience 10

 # TDBRAIN Dataset
 python \
   -u run.py \
   --task_name classification \
   --root_path ./dataset/TDBRAIN/ \
   --model_id TDBRAIN-Indep \
   --model Medformer \
   --data TDBRAIN \
   --e_layers 6 \
   --batch_size 32 \
   --d_model 128 \
   --d_ff 256 \
   --patch_len_list 8,8,8,16,16,16 \
   --augmentations none,drop0.25 \
   --swa \
   --des 'Exp' \
   --itr 5 \
   --learning_rate 0.0001 \
   --train_epochs 100 \
   --patience 10

# ADFTD Dataset
python \
  -u run.py \
  --task_name classification \
  --root_path ./dataset/ADFTD/ \
  --model_id ADFTD-Indep \
  --model Medformer \
  --data ADFTD \
  --e_layers 6 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --patch_len_list 2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32 \
  --augmentations drop0.5 \
  --swa \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

# PTB Dataset
python \
  -u run.py \
  --task_name classification \
  --root_path ./dataset/PTB/ \
  --model_id PTB-Indep \
  --model Medformer \
  --data PTB \
  --e_layers 6 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --patch_len_list 2,4,8,8,16,16,16,32,32,32,32,32 \
  --augmentations none,drop0.5 \
  --swa \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

# PTB-XL Dataset
python \
  -u run.py \
  --task_name classification \
  --root_path ./dataset/PTB-XL/ \
  --model_id PTB-XL-Indep \
  --model Medformer \
  --data PTB-XL \
  --e_layers 6 \
  --batch_size 256 \
  --d_model 128 \
  --d_ff 256 \
  --patch_len_list 2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32 \
  --augmentations jitter0.2,scale0.2,drop0.5 \
  --swa \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10


