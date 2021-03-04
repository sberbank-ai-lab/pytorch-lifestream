# # GRU encoder
# export SC_SUFFIX="encoder_gru"
# python ../../pl_train_module.py \
#     logger_name=${SC_SUFFIX} \
#     trainer.max_epochs=1 \
#     params.rnn.type="gru" \
#     model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
#     --conf conf/mles_params.hocon

# python ../../pl_inference.py \
#     model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
#     output.path="data/emb__$SC_SUFFIX" \
#     --conf conf/mles_params.hocon

# # LSTM encoder
# export SC_SUFFIX="encoder_lstm"
# python ../../pl_train_module.py \
#     logger_name=${SC_SUFFIX} \
#     trainer.max_epochs=1 \
#     params.rnn.type="lstm" \
#     model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
#     --conf conf/mles_params.hocon

# python ../../pl_inference.py \
#     model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
#     output.path="data/emb__$SC_SUFFIX" \
#     --conf conf/mles_params.hocon


# Transformer encoder
export SC_SUFFIX="encoder_transf_bs064_4head_64hs_4layers"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.encoder_type="transf" \
    trainer.max_epochs=1 \
    inference_dataloader.loader.batch_size=1 \
    params.transf.input_size=64 \
    params.transf.shared_layers=false \
    params.transf.use_after_mask=false \
    params.transf.use_src_key_padding_mask=false \
    params.transf.use_positional_encoding=false \
    params.transf.train_starter=true \
    params.transf.dim_hidden=64 \
    params.transf.dropout=0.1 \
    params.transf.max_seq_len=100 \
    params.transf.n_heads=4 \
    params.transf.n_layers=4 \
    params.train.batch_size=32 \
    params.valid.batch_size=32 \
    params.train.split_strategy.cnt_min=50 \
    params.train.split_strategy.cnt_max=200 \
    params.valid.split_strategy.cnt_min=50 \
    params.valid.split_strategy.cnt_max=200 \
    model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
    --conf conf/mles_params.hocon

# python ../../pl_inference.py \
#   model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
#   inference_dataloader.loader.batch_size=128 \
#   output.path="data/emb__$SC_SUFFIX" \
#   --conf conf/mles_params.hocon

# # Transformer encoder
# export SC_SUFFIX="encoder_transf_bs064_4head_64hs_8layers"
# python ../../pl_train_module.py \
#     params.device="$SC_DEVICE" \
#     params.model_type="transf" \
#     params.transf.n_heads=4 \
#     params.transf.n_layers=8 \
#     trainer.max_epochs=1 \
#     params.train.batch_size=64 \
#     params.valid.batch_size=64 \
#     params.train.split_strategy.cnt_min=50 \
#     params.train.split_strategy.cnt_max=200 \
#     params.valid.split_strategy.cnt_min=50 \
#     params.valid.split_strategy.cnt_max=200 \
#     model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
#     --conf conf/mles_params.hocon

# python ../../pl_inference.py \
#     model_path="models/bowl2019_mlm__$SC_SUFFIX.p" \
#     params.valid.batch_size=32 \
#     output.path="data/emb__$SC_SUFFIX" \
#     --conf conf/mles_params.hocon

# # Compare
# rm results/scenario_bowl2019__encoder_types.txt
# # rm -r conf/embeddings_validation.work/
# python -m embeddings_validation \
#     --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
#     --conf_extra \
#       'report_file: "../results/scenario_bowl2019__encoder_types.txt",
#       auto_features: ["../data/emb_mles__encoder_*.pickle"]'



