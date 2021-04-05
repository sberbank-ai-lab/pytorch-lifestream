for is_sorted in "True" "False";do
    for i in 10,30,100 30,60,70 70,100,40 100,150,30; do
        IFS=',' read item1 item2 item3<<< "${i}"
        export cnt_min=${item1}
        export cnt_max=${item2}
        export split_count=${item3}
        export SC_SUFFIX="cpc_v2_sub_seq_sampl_strategy_is_sorted_${is_sorted}_cnt_min_${cnt_min}_cnt_max_${cnt_max}_split_count_${split_count}"
        echo "${SC_SUFFIX}"

        python ../../pl_train_module.py \
            logger_name=${SC_SUFFIX} \
            data_module.train.split_strategy.is_sorted=$is_sorted \
            data_module.train.split_strategy.cnt_min=$cnt_min \
            data_module.train.split_strategy.cnt_max=$cnt_max \
            data_module.train.split_strategy.split_count=$split_count \
            \
            data_module.valid.split_strategy.is_sorted=$is_sorted \
            data_module.valid.split_strategy.cnt_min=$cnt_min \
            data_module.valid.split_strategy.cnt_max=$cnt_max \
            data_module.valid.split_strategy.split_count=$split_count \
            model_path="models/$SC_SUFFIX.p" \
            --conf "conf/cpc_v2_params.hocon"

        python ../../pl_inference.py \
            model_path="models/$SC_SUFFIX.p" \
            output.path="data/emb__$SC_SUFFIX" \
            --conf "conf/cpc_v2_params.hocon"
    done
done

# Compare
rm results/scenario_bowl2019__subseq_smpl_strategy_cpc_v2.txt
python -m embeddings_validation \
    --conf conf/cpc_v2_embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 --local_scheduler \
    --conf_extra \
      'report_file: "../results/scenario_bowl2019__subseq_smpl_strategy_cpc_v2.txt",
      auto_features: ["../data/emb__cpc_v2_sub_seq_sampl_strategy*.pickle"]'
