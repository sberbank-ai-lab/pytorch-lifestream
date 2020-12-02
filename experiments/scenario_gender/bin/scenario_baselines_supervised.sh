# Train a supervised model and save scores to the file
python -m scenario_gender fit_target params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_target_params.json

# Fine tune the MeLES model in supervised mode and save scores to the file
python ../../metric_learning.py \
  params.device="$SC_DEVICE" \
  params.rnn.hidden_size=256 \
  params.train.loss="MarginLoss" params.train.margin=0.2 params.train.beta=0.4 \
  model_path.model="models/mles_model_for_finetuning.p" \
  --conf conf/dataset.hocon conf/mles_params.json
python -m scenario_gender fit_finetuning \
  params.device="$SC_DEVICE" \
  --conf conf/dataset.hocon conf/fit_finetuning_on_mles_params.json

# Fine tune the CPC model in supervised mode and save scores to the file
python -m scenario_gender fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_cpc_params.json

# Fine tune the RTD model in supervised mode and save scores to the file
python -m scenario_gender fit_finetuning params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/fit_finetuning_on_rtd_params.json

# Compare
rm results/scenario_gender_baselines_supervised.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_baselines_supervised.hocon --workers 10 --total_cpu_count 20
