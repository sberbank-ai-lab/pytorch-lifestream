# ReduceLROnPlateau
export SC_SUFFIX="reduce_on_plateau"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.lr_scheduler.ReduceLROnPlateau=true \
    model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# ReduceLROnPlateau x2 epochs
export SC_SUFFIX="reduce_on_plateau_x2epochs"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.lr_scheduler.ReduceLROnPlateau=true \
    params.lr_scheduler.threshold=0.0001 \
    trainer.max_epochs=200 \
    model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"

# CosineAnnealing
export SC_SUFFIX="cosine_annealing"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lr_scheduler.n_epoch=150 \
    params.lr_scheduler.CosineAnnealing=true \
    model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
    --conf "conf/mles_params.hocon"
python ../../pl_inference.py \
    model_path="models/age_pred_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__$SC_SUFFIX" \
    --conf "conf/mles_params.hocon"


Compare
python -m scenario_age_pred compare_approaches --output_file "results/scenario_lr_schedule.csv" \
    --n_workers 5 --models lgb --embedding_file_names \
    "mles_embeddings.pickle"        \
    "emb__reduce_on_plateau.pickle" \
    "emb__reduce_on_plateau_x2epochs.pickle" \
    "emb__cosine_annealing.pickle"
