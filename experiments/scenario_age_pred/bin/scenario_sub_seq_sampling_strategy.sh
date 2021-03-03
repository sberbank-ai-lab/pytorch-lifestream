export SC_SUFFIX="SampleRandom"
export SC_STRATEGY="SampleRandom"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    data_module.train.split_strategy.split_strategy=$SC_STRATEGY \
    data_module.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
    params.train.split_strategy.cnt_min=200 \
    params.train.split_strategy.cnt_max=600 \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"


export SC_SUFFIX="SplitRandom"
export SC_STRATEGY="SplitRandom"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    data_module.train.max_seq_len=600 \
    data_module.valid.max_seq_len=600 \
    data_module.train.split_strategy.split_strategy=$SC_STRATEGY \
    data_module.valid.split_strategy.split_strategy=$SC_STRATEGY \
    model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# Compare
python -m scenario_age_pred compare_approaches --output_file "results/scenario_age_pred__subseq_smpl_strategy.csv" \
    --models "lgb" --embedding_file_names \
    "mles_embeddings.pickle"        \
    "emb__SplitRandom.pickle" \
    "emb__SampleRandom.pickle"
