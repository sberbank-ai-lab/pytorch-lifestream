# Prepare agg feature encoder and take embedidngs; inference
python ../../pl_train_module.py --conf conf/agg_features_params.hocon
python ../../pl_inference.py    --conf conf/agg_features_params.hocon

# Train the MeLES encoder and take embedidngs; inference
python ../../pl_train_module.py --conf conf/mles_params.hocon
python ../../pl_inference.py    --conf conf/mles_params.hocon

# Train the Contrastive Predictive Coding (CPC) model; inference
python ../../pl_train_module.py --conf conf/cpc_params.hocon
python ../../pl_inference.py    --conf conf/cpc_params.hocon

# Train the Sequence Order Prediction (SOP) model; inference
python ../../train_sop.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/sop_params.json
python ../../ml_inference.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/sop_params.json

# Train the Next Sequence Prediction (NSP) model; inference
python ../../train_nsp.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/nsp_params.json
python ../../ml_inference.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/nsp_params.json

# Train the Replaced Token Detection (RTD) model; inference
python ../../train_rtd.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/rtd_params.json
python ../../ml_inference.py params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/rtd_params.json

# Compare
rm results/scenario_gender_baselines_unsupervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_baselines_unsupervised.hocon --workers 10 --total_cpu_count 20
