## APAVA Dataset
 python \
   -u run.py \
   --task_name classification \
   --root_path ./dataset/APAVA/ \
   --model_id APAVA-Indep \
   --model TimesNet \
   --data APAVA \
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

 # TDBRAIN Dataset
 python \
   -u run.py \
   --task_name classification \
   --root_path ./dataset/TDBRAIN/ \
   --model_id TDBRAIN-Indep \
   --data TDBRAIN \
   --model TimesNet \
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

 #ADFTD Dataset
 python \
   -u run.py \
   --task_name classification \
   --root_path ./dataset/ADFTD/ \
   --model_id ADFTD-Indep \
   --model TimesNet \
   --data ADFTD \
   --e_layers 2 \
   --batch_size 128 \
   --d_model 128 \
   --d_ff 256 \
   --top_k 3 \
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
  --model TimesNet \
  --data PTB \
  --e_layers 2 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --patience 10

# PTB-XL Dataset
python \
  -u run.py \
  --task_name classification \
  --root_path ./dataset/PTB-XL/ \
  --model_id PTB-XL-Indep \
  --model TimesNet \
  --data PTB-XL \
  --e_layers 2 \
  --batch_size 256 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 30 \
  --patience 10





