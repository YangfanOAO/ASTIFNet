
#SYNTHETIC Dataset
python \
 -u run.py \
 --task_name addition \
 --root_path ./dataset/SYNTHETIC/ \
 --model_id SYNTHETIC \
 --model ASTIFNet \
 --data SYNTHETIC \
 --emb_size 8 9 8 \
 --kernel_t 4 5 4 \
 --kernel_s 2 2 2 \
 --des 'Exp' \
 --batch_size 32 \
 --itr 5 \
 --learning_rate 0.002 \
 --lradj 'type3' \
 --train_epochs 100 \
 --patience 10


# UCI-HAR Dataset
python \
 -u run.py \
 --task_name addition \
 --root_path ./dataset/UCI-HAR/ \
 --model_id UCI-HAR \
 --model ASTIFNet \
 --data UCI-HAR \
 --emb_size 12 18 12 \
 --kernel_t 4 5 4 \
 --kernel_s 3 4 3 \
 --des 'Exp' \
 --batch_size 32 \
 --itr 5 \
 --learning_rate 0.002 \
 --lradj 'type3' \
 --train_epochs 100 \
 --patience 10


# FLAAP Dataset
python \
 -u run.py \
 --task_name addition \
 --root_path ./dataset/FLAAP/ \
 --model_id FLAAP \
 --model ASTIFNet \
 --data FLAAP \
 --emb_size 24 24 24 \
 --kernel_t 3 4 3 \
 --kernel_s 2 2 2 \
 --des 'Exp' \
 --batch_size 32 \
 --itr 5 \
 --learning_rate 0.002 \
 --lradj 'type3' \
 --train_epochs 100 \
 --patience 10
