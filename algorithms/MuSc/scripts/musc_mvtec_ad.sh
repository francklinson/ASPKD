# MVTec AD
python3 musc_main.py --device 0 \
--data_path ../data/mvtec/  --dataset_name mvtec_ad --class_name bottle \
--backbone_name ViT-B-32 --pretrained openai --feature_layers 2 5 8 11 \
--img_resize 512 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True


python musc_main.py --device 0 \
--data_path ../data/spk/  --dataset_name mvtec_ad --class_name qzgy \
--backbone_name ViT-B-32 --pretrained openai --feature_layers 2 5 8 11 \
--img_resize 512 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True


# SPK
python musc_main.py --device 0 \
--data_path ../data/spk/  --dataset_name mvtec_ad --class_name N32 \
--backbone_name ViT-B-32 --pretrained openai --feature_layers 2 5 8 11 \
--img_resize 256 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True



python musc_main.py --device 0 \
--data_path ../data/mvtec/  --dataset_name mvtec_ad --class_name bottle \
--backbone_name ViT-B-32 --pretrained openai --feature_layers 2 5 8 11 \
--img_resize 256 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True



python musc_main.py --device 0 \
--data_path ../data/mvtec/  --dataset_name mvtec_ad --class_name bottle \
--backbone_name ViT-B-16 --pretrained openai --feature_layers 2 5 8 11 \
--img_resize 512 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True




python musc_main.py --device 0 \
--data_path ../data/mvtec/  --dataset_name mvtec_ad --class_name bottle \
--backbone_name ViT-B-16 --pretrained openai --feature_layers 2 5 8 11 \
--img_resize 256 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True




python musc_main.py --device 0 \
--data_path ../data/mvtec/  --dataset_name mvtec_ad --class_name bottle \
--backbone_name ViT-B-16-plus-240 --pretrained laion400m_e31 --feature_layers 2 5 8 11 \
--img_resize 512 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True



python musc_main.py --device 1 \
--data_path ./data/mvtec_anomaly_detection/ --dataset_name mvtec_ad --class_name ALL \
--backbone_name ViT-B-16-plus-240 --pretrained laion400m_e31 --feature_layers 2 5 8 11 \
--img_resize 240 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis False --save_excel True



python musc_main.py --device 0 \
--data_path ../data/mvtec/  --dataset_name mvtec_ad --class_name bottle \
--backbone_name ViT-L-14 --pretrained openai --feature_layers 5 11 17 23 \
--img_resize 518 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True



python musc_main.py --device 0 \
--data_path ./data/mvtec_anomaly_detection/ --dataset_name mvtec_ad --class_name ALL \
--backbone_name ViT-L-14 --pretrained openai --feature_layers 5 11 17 23 \
--img_resize 336 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis False --save_excel True



python musc_main.py --device 1 \
--data_path ./data/mvtec_anomaly_detection/ --dataset_name mvtec_ad --class_name ALL \
--backbone_name ViT-L-14-336 --pretrained openai --feature_layers 5 11 17 23 \
--img_resize 518 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis False --save_excel True



python musc_main.py --device 1 \
--data_path ./data/mvtec_anomaly_detection/ --dataset_name mvtec_ad --class_name ALL \
--backbone_name ViT-L-14-336 --pretrained openai --feature_layers 5 11 17 23 \
--img_resize 336 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis False --save_excel True


python musc_main.py --device 0 \
--data_path ../data/mvtec/  --dataset_name mvtec_ad --class_name bottle \
--backbone_name dino_vitbase16 --pretrained laion400m_e31 --feature_layers 2 5 8 11 \
--img_resize 512 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True


python musc_main.py --device 0 \
--data_path ../data/mvtec/  --dataset_name mvtec_ad --class_name bottle \
--backbone_name dino_vitbase16 --pretrained laion400m_e31 --feature_layers 2 5 8 11 \
--img_resize 256 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True


python musc_main.py --device 0 \
--data_path ../data/mvtec/  --dataset_name mvtec_ad --class_name bottle \
--backbone_name dinov2_vitb14 --pretrained laion400m_e31 --feature_layers 2 5 8 11 \
--img_resize 518 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True



python musc_main.py --device 0 \
--data_path ../data/mvtec/  --dataset_name mvtec_ad --class_name bottle \
--backbone_name dinov2_vitb14 --pretrained laion400m_e31 --feature_layers 2 5 8 11 \
--img_resize 336 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True


python musc_main.py --device 0 \
--data_path ../data/mvtec/  --dataset_name mvtec_ad --class_name bottle \
--backbone_name dinov2_vitl14 --pretrained laion400m_e31 --feature_layers 5 11 17 23 \
--img_resize 518 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True


python musc_main.py --device 0 \
--data_path ../data/spk/  --dataset_name mvtec_ad --class_name TRY \
--backbone_name dinov2_vitl14 --pretrained laion400m_e31 --feature_layers 5 11 17 23 \
--img_resize 518 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True




python musc_main.py --device 0 \
--data_path ../data/mvtec/  --dataset_name mvtec_ad --class_name bottle \
--backbone_name dinov2_vitl14 --pretrained laion400m_e31 --feature_layers 5 11 17 23 \
--img_resize 336 --divide_num 1 --r_list 1 3 5 --batch_size 4 \
--output_dir ./output --vis True --save_excel True
