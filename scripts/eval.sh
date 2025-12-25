export CUDA_VISIBLE_DEVICES=3
python unigeoseg/eval_and_test/eval.py \
  --model_path /data1/ns/pyprj/earthmindrsrs/SegEarth-R1/checkpoint/SegEarth-R1_ablation_rand_dynamic_sampling_allrrsisdx5_bs2x8_lr1e4_ep3_SparseConv1_mte_true512_latent4_latent02_nointera_enhancedreasonpool_1018/checkpoint-221460 \
  --data_split "test" \
  --version "llava_phi" \
  --mask_config "unigeoseg/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml" \
  --dataset_type "RSRS" \
  --base_data_path "/data1/ns/RemoteSensingReasonSeg/benchmark/ref/test" \
