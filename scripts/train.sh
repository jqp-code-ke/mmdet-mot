set CUDA_VISIBLE_DEVICES=0 & python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 .\tools\train.py configs --launcher pytorch --no-validate
