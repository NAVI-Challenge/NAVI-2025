job_name="navi_train"
DATA_PATH="path/to/data_navi folder/"

export NCCL_TIMEOUT=3600
export MASTER_PORT=30001


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 \
    main_xclip.py --num_thread_reader=8 \
    --do_train \
    --epochs=40 --batch_size=40 --batch_size_val=40 --gradient_accumulation_steps=1 --n_display=40 --n_gpu=4 \
    --data_path ${DATA_PATH}/navi/input/ \
    --features_path ${DATA_PATH}/navi/json/ \
    --output_dir ${DATA_PATH}/navi/output/ \
    --frame_path "path/to/DATA folder" \
    --max_words 256 --max_frames 20 \
    --datatype navi --train_model_from "" \
    --feature_framerate 1 --lr 1e-3 --coef_lr 1e-3 \
    --freeze_layer_num 0 --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/16 \
    --init_model "path/to/pretrained/model/bin" 2>&1 | tee -a ${DATA_PATH}/navi/output/log/${job_name}.log

