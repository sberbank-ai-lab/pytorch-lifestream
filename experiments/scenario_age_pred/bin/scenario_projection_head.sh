for PRJ_SIZE in 256 128 064
do
    for RNN_SIZE in 1600 0800 0400
    do
        export SC_SUFFIX="projection_head_rnn${RNN_SIZE}_prh${PRJ_SIZE}"
        python ../../pl_train_module.py \
            logger_name=${SC_SUFFIX} \
            params.rnn.hidden_size=${RNN_SIZE} \
            "params.head_layers=[[Linear, {in_features: ${RNN_SIZE}, out_features: ${PRJ_SIZE}}], [BatchNorm1d, {num_features: ${PRJ_SIZE}}], [ReLU, {}], [Linear, {in_features: ${PRJ_SIZE}, out_features: ${PRJ_SIZE}}], [NormEncoder, {}]]" \
            model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
            --conf conf/mles_proj_head_params.hocon
        python ../../pl_inference.py \
            model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
            output.path="data/emb__$SC_SUFFIX" \
            --conf conf/mles_proj_head_params.hocon
    done
done

# Compare
python -m scenario_age_pred compare_approaches --output_file "results/scenario_age_pred__proj_head.csv" \
    --embedding_file_names \
    "emb__projection_head_rnn1600_prh256.pickle" \
    "emb__projection_head_rnn0800_prh256.pickle" \
    "emb__projection_head_rnn0400_prh256.pickle" \
    "emb__projection_head_rnn1600_prh128.pickle" \
    "emb__projection_head_rnn0800_prh128.pickle" \
    "emb__projection_head_rnn0400_prh128.pickle" \
    "emb__projection_head_rnn1600_prh064.pickle" \
    "emb__projection_head_rnn0800_prh064.pickle" \
    "emb__projection_head_rnn0400_prh064.pickle"
