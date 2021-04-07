# # # Prepare agg feature encoder and take embedidngs; inference
python ../../pl_train_module.py --conf conf/agg_features_params.hocon
python ../../pl_inference.py --conf conf/agg_features_params.hocon

# Train the MeLES encoder and take embedidngs; inference
python ../../pl_train_module.py --conf conf/mles_params.hocon
python ../../pl_inference.py    --conf conf/mles_params.hocon

# Train the Contrastive Predictive Coding (CPC) model; inference
python ../../pl_train_module.py --conf conf/cpc_params.hocon
python ../../pl_inference.py    --conf conf/cpc_params.hocon

# # Train the Contrastive Predictive Coding (CPC_V2) model; inference
python ../../pl_train_module.py s--conf conf/cpc_v2_params.hocon
python ../../pl_inference.py --conf conf/cpc_v2_params.hocon


# # Compare
rm results/scenario_gender_baselines_unsupervised_cpc_v2.txt
# rm -r conf/embeddings_validation.work/

python -m embeddings_validation \
    --conf conf/cpc_v2_embeddings_validation_baselines_unsupervised.hocon --workers 10 --total_cpu_count 20 --local_scheduler
