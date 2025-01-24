 #APAVA Dataset
 python \
  -u run.py \
  --task_name classification \
  --root_path ./dataset/APAVA/ \
  --model_id APAVA-Indep \
  --model ASTIFNet \
  --data APAVA \
  --emb_size 8 9 8 \
  --kernel_t 4 5 4 \
  --kernel_s 3 4 3 \
  --des 'Exp' \
  --batch_size 32 \
  --itr 5 \
  --learning_rate 0.002 \
  --lradj 'type3' \
  --train_epochs 100 \
  --patience 10


# TDBRAIN Dataset
 python \
   -u run.py \
   --task_name classification \
   --root_path ./dataset/TDBRAIN/ \
   --model_id TDBRAIN-Indep \
   --model ASTIFNet \
   --data TDBRAIN \
   --emb_size 7 8 9 \
   --kernel_t 5 6 5 \
   --kernel_s 4 3 4 \
   --des 'Exp' \
   --batch_size 32 \
   --itr 5 \
   --learning_rate 0.002 \
   --lradj 'type3' \
   --train_epochs 100 \
   --patience 10


## ADFTD Dataset
   python \
     -u run.py \
     --task_name classification \
     --root_path ./dataset/ADFTD/ \
     --model_id ADFTD-Indep \
     --model ASTIFNet \
     --data ADFTD \
     --emb_size 9 11 9 \
     --kernel_t 5 4 5 \
     --kernel_s 3 3 3 \
     --des 'Exp' \
     --itr 5 \
     --batch_size 128 \
     --learning_rate 0.002 \
     --lradj 'type3' \
     --train_epochs 100 \
     --patience 10


# PTB Dataset
  python \
    -u run.py \
    --task_name classification \
    --root_path ./dataset/PTB/ \
    --model_id PTB-Indep \
    --model ASTIFNet \
    --data PTB \
    --emb_size 11 18 11 \
    --kernel_t 7 4 5 \
    --kernel_s 4 5 4 \
    --des 'Exp' \
    --itr 5 \
    --batch_size 128 \
    --learning_rate 0.002 \
    --lradj 'type3' \
    --train_epochs 100 \
    --patience 10



## PTB-XL Dataset
python \
 -u run.py \
 --task_name classification \
 --root_path ./dataset/PTB-XL/ \
 --model_id PTB-XL-Indep \
 --model ASTIFNet \
 --data PTB-XL \
 --emb_size 18 18 18 \
 --kernel_t 7 5 7 \
 --kernel_s 3 4 3 \
 --des 'Exp' \
 --itr 5 \
 --batch_size 256 \
 --learning_rate 0.002 \
 --lradj 'type3' \
 --train_epochs 100 \
 --patience 10
