# APAVA Dataset
python -u run.py \
  --task_name classification \
  --root_path ./dataset/APAVA/ \
  --model_id APAVA-Indep \
  --data APAVA \
  --model ModernTCN \
  --batch_size 32 \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 31 19 \
  --small_size 5 5 \
  --dims 128 256 \
  --head_dropout 0.0 \
  --class_dropout 0.0 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10


# TDBRAIN Dataset
python -u run.py \
  --task_name classification \
  --root_path ./dataset/TDBRAIN/ \
  --model_id TDBRAIN-Indep \
  --data TDBRAIN \
  --model ModernTCN \
  --batch_size 32 \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 31 19 \
  --small_size 5 5 \
  --dims 128 256 \
  --head_dropout 0.0 \
  --class_dropout 0.0 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

# ADFTD Dataset
python -u run.py \
  --task_name classification \
  --root_path ./dataset/ADFTD/ \
  --model_id ADFTD-Indep \
  --data ADFTD \
  --model ModernTCN \
  --batch_size 128 \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 31 19 \
  --small_size 5 5 \
  --dims 128 256 \
  --head_dropout 0.0 \
  --class_dropout 0.0 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

# PTB Dataset
python -u run.py \
  --task_name classification \
  --root_path ./dataset/PTB/ \
  --model_id PTB-Indep \
  --data PTB \
  --model ModernTCN \
  --batch_size 128 \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 31 19 \
  --small_size 5 5 \
  --dims 128 256 \
  --head_dropout 0.0 \
  --class_dropout 0.0 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10

# PTB-XL Dataset

python -u run.py \
  --task_name classification \
  --root_path ./dataset/PTB-XL/ \
  --model_id PTB-XL-Indep \
  --data PTB-XL \
  --model ModernTCN \
  --batch_size 256 \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 1 \
  --large_size 31 19 \
  --small_size 5 5 \
  --dims 128 256 \
  --head_dropout 0.0 \
  --class_dropout 0.0 \
  --dropout 0.1 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10