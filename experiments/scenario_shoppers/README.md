# Get data

```sh
cd experiments/scenario_shoppers

# download datasets
sh bin/get-data.sh

# convert datasets from transaction list to features for metric learning
sh bin/make-datasets-spark.sh
```

# Main scenario, best params

```sh
cd experiments/scenario_shoppers
export CUDA_VISIBLE_DEVICES=0  # define here one gpu device number

sh bin/run_all_scenarios.sh

# check the results
cat results/*.txt
cat results/*.csv
```
