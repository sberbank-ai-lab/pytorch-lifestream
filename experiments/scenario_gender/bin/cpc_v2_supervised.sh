# Train a supervised model and save scores to the file
# python ../../pl_fit_target.py trainer.max_epochs=2 --conf conf/pl_fit_target.hocon

# Fine tune the CPC model in supervised mode and save scores to the file
# python ../../pl_fit_target.py --conf conf/cpc_v2_pl_fit_finetuning.hocon


# Compare
# rm results/scenario_gender_baselines_supervised_cpc_v2.txt
# rm -r conf/embeddings_validation.work/
# python -m embeddings_validation \
#     --conf conf/cpc_v2_embeddings_validation_baselines_supervised.hocon --workers 10 --total_cpu_count 20 --local_scheduler