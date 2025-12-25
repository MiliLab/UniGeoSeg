export CUDA_VISIBLE_DEVICES=3
python unigeoseg/eval_and_test/eval.py \
  --model_path /path/to/checkpoint \
  --data_split "test" \
  --version "llava_phi" \
  --mask_config "unigeoseg/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml" \
  --dataset_type "RSRS" \
  --base_data_path "UniGeoSeg/benchmark/ref/test" \
