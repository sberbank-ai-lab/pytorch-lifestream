export SC_SUFFIX="bt_tuning_base"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.9025 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_hidden_size_0680"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=680 \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.9025 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_batch_size_96"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    data_module.train.batch_size=96 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.9025 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_lambd_0.02"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.02 \
    params.rnn.hidden_size=1024 \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.9025 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=500 \
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    output.path="data/emb__${SC_SUFFIX}" \
    --conf "conf/barlow_twins_params.hocon"


#################

export SC_SUFFIX="bt_tuning_v01"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[Linear, {in_features: 1024, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256}], [ReLU, {}], [Linear, {in_features: 256, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_v02"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[Linear, {in_features: 1024, out_features: 128, bias: false}], [BatchNorm1d, {num_features: 128}], [ReLU, {}], [Linear, {in_features: 128, out_features: 128, bias: false}], [BatchNorm1d, {num_features: 128, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_v03"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[Linear, {in_features: 1024, out_features: 512, bias: false}], [BatchNorm1d, {num_features: 512}], [ReLU, {}], [Linear, {in_features: 512, out_features: 512, bias: false}], [BatchNorm1d, {num_features: 512, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.001 \
    params.train.weight_decay=0.001 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_v04"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.02 \
    params.rnn.hidden_size=2048 \
    "params.head_layers=[[Linear, {in_features: 2048, out_features: 512, bias: false}], [BatchNorm1d, {num_features: 512}], [ReLU, {}], [Linear, {in_features: 512, out_features: 512, bias: false}], [BatchNorm1d, {num_features: 512, affine: False}]]" \
    data_module.train.batch_size=256 \
    params.train.lr=0.002 \
    params.train.weight_decay=0.001 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.9025 \
    trainer.max_epochs=400 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_v05"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.01 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[Linear, {in_features: 1024, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256}], [ReLU, {}], [Linear, {in_features: 256, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.7 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_v06"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.02 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[Linear, {in_features: 1024, out_features: 128, bias: false}], [BatchNorm1d, {num_features: 128}], [ReLU, {}], [Linear, {in_features: 128, out_features: 128, bias: false}], [BatchNorm1d, {num_features: 128, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.7 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_v07"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[BatchNorm1d, {num_features: 1024}], [ReLU, {}], [Linear, {in_features: 1024, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_v08"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[Dropout, {p: 0.1}], [Linear, {in_features: 1024, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_v09"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.08 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[ReLU, {}], [Linear, {in_features: 1024, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_v10"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.002 \
    params.rnn.hidden_size=1024 \
    "params.head_layers=[[ReLU, {}], [Linear, {in_features: 1024, out_features: 256, bias: false}], [BatchNorm1d, {num_features: 256, affine: False}]]" \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"


export SC_SUFFIX="bt_tuning_v11"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_v12"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.04 \
    params.rnn.hidden_size=1024 \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"

export SC_SUFFIX="bt_tuning_v13"
python ../../pl_train_module.py \
    logger_name=${SC_SUFFIX} \
    params.train.lambd=0.002 \
    params.rnn.hidden_size=1024 \
    data_module.train.batch_size=128 \
    params.train.lr=0.004 \
    params.train.weight_decay=0 \
    params.lr_scheduler.step_size=10 \
    params.lr_scheduler.step_gamma=0.9025 \
    trainer.max_epochs=300 \
    params.train.checkpoints_every_n_val_epochs=10 trainer.checkpoint_callback=none\
    model_path="models/gender_mlm__$SC_SUFFIX.p" \
    --conf "conf/barlow_twins_params.hocon"


export SC_SUFFIX="bt_tuning_v11"
export SC_VERSION=0
export SC_SUFFIX="bt_tuning_v12"
export SC_VERSION=1
export SC_SUFFIX="bt_tuning_v13"
export SC_VERSION=0



python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=9-step\=589.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_009" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=19-step\=1179.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_019" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=29-step\=1769.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_029" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=39-step\=2359.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_039" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=49-step\=2949.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_049" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=59-step\=3539.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_059" \
    --conf "conf/barlow_twins_params.hocon"

python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=69-step\=4129.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_069" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=79-step\=4719.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_079" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=89-step\=5309.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_089" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=99-step\=5899.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_099" \
    --conf "conf/barlow_twins_params.hocon"

python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=109-step\=6489.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_109" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=119-step\=7079.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_119" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=129-step\=7669.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_129" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=139-step\=8259.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_139" \
    --conf "conf/barlow_twins_params.hocon"

python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=149-step\=8849.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_149" \
    --conf "conf/barlow_twins_params.hocon"

python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=199-step\=11799.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_199" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=249-step\=14749.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_249" \
    --conf "conf/barlow_twins_params.hocon"
python ../../pl_inference.py     inference_dataloader.loader.batch_size=200 \
    model_path="lightning_logs/${SC_SUFFIX}/version_${SC_VERSION}/checkpoints/epoch\=299-step\=17699.ckpt" \
    output.path="data/emb__${SC_SUFFIX}_299" \
    --conf "conf/barlow_twins_params.hocon"


rm results/res_bt_tuning.txt
# rm -r conf/embeddings_validation.work/
python -m embeddings_validation \
    --conf conf/embeddings_validation_short.hocon --workers 10 --total_cpu_count 20 \
    --conf_extra \
      'report_file: "../results/res_bt_tuning.txt",
      auto_features: ["../data/emb__bt_tuning_*.pickle", "../data/barlow_twins_embeddings.pickle"]'
less -S results/res_bt_tuning.txt




###########
# Valid auroc     : 0.789 [0.790, 0.784, 0.797, 0.788, 0.788], Test  auroc     : 0.827 [0.817, 0.829, 0.830, 0.829, 0.830]
python ../../pl_fit_target.py \
  logger_name="bt_ftning_base" \
  --conf conf/pl_fit_finetuning_barlow_twins.hocon

# Valid auroc     : 0.786 [0.788, 0.779, 0.792, 0.786, 0.787], Test  auroc     : 0.827 [0.817, 0.828, 0.830, 0.829, 0.829]
python ../../pl_fit_target.py \
  logger_name="bt_ftning_v01" \
  params.train.lr=0.005 \
  --conf conf/pl_fit_finetuning_barlow_twins.hocon

# Valid auroc     : 0.787 [0.788, 0.779, 0.794, 0.787, 0.787], Test  auroc     : 0.824 [0.814, 0.825, 0.828, 0.826, 0.827]
python ../../pl_fit_target.py \
  logger_name="bt_ftning_v02" \
  params.lr_scheduler.step_size=2 \
  --conf conf/pl_fit_finetuning_barlow_twins.hocon

# Valid auroc     : 0.788 [0.784, 0.776, 0.791, 0.790, 0.798], Test  auroc     : 0.812 [0.801, 0.811, 0.817, 0.815, 0.817]
#  seq_enc lr=0.001
# more head 256
python ../../pl_fit_target.py \
  logger_name="bt_ftning_v05" \
  --conf conf/pl_fit_finetuning_barlow_twins.hocon



# Valid auroc     : 0.788 [0.789, 0.784, 0.796, 0.787, 0.786], Test  auroc     : 0.827 [0.819, 0.829, 0.830, 0.829, 0.831]
# freeze seq_enc
python ../../pl_fit_target.py \
  logger_name="bt_ftning_v03" \
  --conf conf/pl_fit_finetuning_barlow_twins.hocon

# Valid auroc     : 0.788 [0.787, 0.776, 0.792, 0.791, 0.793], Test  auroc     : 0.821 [0.815, 0.823, 0.822, 0.826, 0.820]
# freeze seq_enc
# more head 256
python ../../pl_fit_target.py \
  logger_name="bt_ftning_v04" \
  --conf conf/pl_fit_finetuning_barlow_twins.hocon

# Valid auroc     : 0.789 [0.789, 0.778, 0.796, 0.788, 0.793], Test  auroc     : 0.825 [0.817, 0.826, 0.825, 0.827, 0.829]
# freeze seq_enc
# more head 64
python ../../pl_fit_target.py \
  logger_name="bt_ftning_v07" \
  --conf conf/pl_fit_finetuning_barlow_twins.hocon


# Valid auroc     : 0.790 [0.791, 0.780, 0.797, 0.790, 0.794], Test  auroc     : 0.827 [0.820, 0.827, 0.830, 0.828, 0.830]
# freeze seq_enc
# more head 64
# minus first bn
python ../../pl_fit_target.py \
  logger_name="bt_ftning_v07" \
  --conf conf/pl_fit_finetuning_barlow_twins.hocon


# Valid auroc     : 0.792 [0.792, 0.781, 0.797, 0.792, 0.796], Test  auroc     : 0.826 [0.818, 0.826, 0.828, 0.826, 0.830]
# seq_enc  lr: 0.0001
# more head 64
# minus first bn
python ../../pl_fit_target.py \
  logger_name="bt_ftning_v08" \
  --conf conf/pl_fit_finetuning_barlow_twins.hocon

# Valid auroc     : 0.789 [0.790, 0.779, 0.795, 0.787, 0.793], Test  auroc     : 0.823 [0.817, 0.820, 0.825, 0.823, 0.828]
# seq_enc  lr: 0.0001
# more head 64-8
# minus first bn
python ../../pl_fit_target.py \
  logger_name="bt_ftning_v09" \
  --conf conf/pl_fit_finetuning_barlow_twins.hocon

# Valid auroc     : 0.788 [0.793, 0.781, 0.792, 0.784, 0.791], Test  auroc     : 0.824 [0.816, 0.826, 0.824, 0.827, 0.829]
# seq_enc  lr: 0.0001
# more head 64-8
# minus first bn
python ../../pl_fit_target.py \
  logger_name="bt_ftning_v10" \
  --conf conf/pl_fit_finetuning_barlow_twins.hocon


# Valid auroc     : 0.791 [0.791, 0.780, 0.797, 0.791, 0.794], Test  auroc     : 0.827 [0.819, 0.827, 0.828, 0.828, 0.831]
# seq_enc  lr: 0.00001
# more head 64
# minus first bn
python ../../pl_fit_target.py \
  logger_name="bt_ftning_v11" \
  --conf conf/pl_fit_finetuning_barlow_twins.hocon


#
# seq_enc  lr: 0.00001
# more head 64
# minus first bn
python ../../pl_fit_target.py \
  trainer.max_epochs=10 \
  logger_name="bt_ftning_v12" \
  --conf conf/pl_fit_finetuning_barlow_twins.hocon

#
# seq_enc  lr: 0.00001
# more head 64
# minus first bn
python ../../pl_fit_target.py \
  trainer.max_epochs=6 \
  logger_name="bt_ftning_v13" \
  --conf conf/pl_fit_finetuning_barlow_twins.hocon
