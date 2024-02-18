inference_single_image(){
input_rgb_path="/data/shuozhang/dataset/casia/val_img"
dataset_name='casia'
output_dir='../testevalute'
pretrained_model_path="/data/shuozhang/forgery_detection/replace_vae_with_backbone_outputs/checkpoint-40013"
ensemble_size=10

cd ..
cd Inference

CUDA_VISIBLE_DEVICES=6 python run_inference.py \
    --input_rgb_path $input_rgb_path \
    --dataset_name $dataset_name \
    --output_dir $output_dir \
    --pretrained_model_path $pretrained_model_path \
    --ensemble_size $ensemble_size \
    --replace_vae_with_backbone \

}

inference_single_image
