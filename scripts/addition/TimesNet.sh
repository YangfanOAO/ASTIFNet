# SYNTHETIC Dataset
 python \
   -u run.py \
   --task_name addition \
   --root_path ./dataset/SYNTHETIC/ \
   --model_id SYNTHETIC \
   --model TimesNet \
   --data SYNTHETIC \
   --e_layers 2 \
   --batch_size 32 \
   --d_model 128 \
   --d_ff 256 \
   --top_k 3 \
   --des 'Exp' \
   --itr 5 \
   --learning_rate 0.0001 \
   --train_epochs 100 \
   --patience 10

# UCI-HAR Dataset
 python \
   -u run.py \
   --task_name addition \
   --root_path ./dataset/UCI-HAR/ \
   --model_id UCI-HAR \
   --model TimesNet \
   --data UCI-HAR \
   --e_layers 2 \
   --batch_size 32 \
   --d_model 128 \
   --d_ff 256 \
   --top_k 3 \
   --des 'Exp' \
   --itr 5 \
   --learning_rate 0.0001 \
   --train_epochs 100 \
   --patience 10


# FLAAP Dataset
 python \
   -u run.py \
   --task_name addition \
   --root_path ./dataset/FLAAP/ \
   --model_id FLAAP \
   --model TimesNet \
   --data FLAAP \
   --e_layers 2 \
   --batch_size 32 \
   --d_model 128 \
   --d_ff 256 \
   --top_k 3 \
   --des 'Exp' \
   --itr 5 \
   --learning_rate 0.0001 \
   --train_epochs 100 \
   --patience 10






