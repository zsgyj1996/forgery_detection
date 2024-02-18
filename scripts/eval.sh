eval_single_image(){
pred_dir="/data/shuozhang/forgery_detection/testevalute/colored"
gt_file="/data/shuozhang/dataset/casia/val_gt"


cd ..
cd evaluate

CUDA_VISIBLE_DEVICES=1 python evaluate.py \
    --pred_dir $pred_dir \
    --gt_file $gt_file \

}

eval_single_image
