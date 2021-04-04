# for is_sorted in "True" "False"
# do
#     for item in 
# done
# for i in 10,20,40 20,50,30 50,80,40; do
# for i in 10,20,40; do
#     IFS=',' read item1 item2 item3<<< "${i}"
#     export cnt_min=${item1}
#     export cnt_max=${item2}
#     export split_count=${item3}
#     export SC_SUFFIX="cpc_v2_sub_seq_sampl_strategy_cnt_min_${cnt_min}_cnt_max_${cnt_max}_split_count_${split_count}"
#     echo "${SC_SUFFIX}"

#     python ../../pl_train_module.py \
#         logger_name=${SC_SUFFIX} \
#         trainer.max_epochs=1 \
#         data_module.train.split_strategy.cnt_min=$cnt_min \
#         data_module.train.split_strategy.cnt_max=$cnt_max \
#         data_module.train.split_strategy.split_count=$split_count \
#         \
#         data_module.valid.split_strategy.cnt_min=$cnt_min \
#         data_module.valid.split_strategy.cnt_max=$cnt_max \
#         data_module.valid.split_strategy.split_count=$split_count \
#         model_path="models/$SC_SUFFIX.p" \
#         --conf "conf/cpc_v2_params.hocon"

#     python ../../pl_inference.py \
#         model_path="models/$SC_SUFFIX.p" \
#         output.path="data/emb__$SC_SUFFIX" \
#         --conf "conf/cpc_v2_params.hocon"
# done

# Compare
rm results/scenario_bowl2019__subseq_smpl_strategy_cpc_v2.txt
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 --local_scheduler \
    --conf_extra \
      'report_file: "../results/scenario_bowl2019__subseq_smpl_strategy_cpc_v2.txt",
      auto_features: ["../data/emb__cpc_v2_sub_seq_sampl_strategy*.pickle"]'

