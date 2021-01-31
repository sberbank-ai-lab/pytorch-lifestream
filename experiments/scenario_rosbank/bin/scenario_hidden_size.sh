for SC_HIDDEN_SIZE in 3072 2048 1024 0512 0256 0128 0064 0032
do
  export SC_SUFFIX="hidden_size_${SC_HIDDEN_SIZE}"
  python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.rnn.hidden_size=${SC_HIDDEN_SIZE} \
    model_path="models/mles__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
  
  python ../../pl_inference.py \
    model_path="models/mles__$SC_SUFFIX.p" \
    output.path="data/emb_mles__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"
done