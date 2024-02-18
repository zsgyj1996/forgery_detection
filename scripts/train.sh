LAUNCH_TRAINING(){

# accelerate config default
cd .. 
cd training
pretrained_model_name_or_path='/data/huangjiaming/stable-diffusion-2'
dataset_path='/data/shuozhang/merged/dataset'
dataset_name='casia'
trainlist='/data/shuozhang/merged/dataset/tp_list.txt'
val_img='/data/shuozhang/merged/dataset/val_img'
val_gt='/data/shuozhang/merged/dataset/val_gt'
output_dir='../not_use_vae_rgb_mergered'
train_batch_size=4
num_train_epochs=50000
gradient_accumulation_steps=2
learning_rate=3e-5
lr_warmup_steps=0
dataloader_num_workers=4
tracker_project_name='forgery-detection'
multires_noise_iterations=6
main_process_port=20547

CUDA_VISIBLE_DEVICES=0,2,3,5 accelerate launch --mixed_precision="fp16"  --multi_gpu --main_process_port $main_process_port run_train.py \
                  --pretrained_model_name_or_path $pretrained_model_name_or_path \
                  --dataset_name  $dataset_name --trainlist $trainlist \
                  --dataset_path $dataset_path --val_img $val_img \
                  --output_dir $output_dir --val_gt $val_gt \
                  --train_batch_size $train_batch_size \
                  --num_train_epochs $num_train_epochs \
                  --gradient_accumulation_steps $gradient_accumulation_steps\
                  --learning_rate $learning_rate \
                  --lr_warmup_steps $lr_warmup_steps \
                  --dataloader_num_workers $dataloader_num_workers \
                  --tracker_project_name $tracker_project_name \
                  --enable_xformers_memory_efficient_attention \
                  --multires_noise_iterations $multires_noise_iterations \
                  --not_use_vae_rgb \


}

LAUNCH_TRAINING