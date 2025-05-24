#!/bin/bash
env_name=biology-simulations
cmd_name=export_conda_env
export_path=$(dirname $(dirname $(readlink -f $0)))/environment.yaml
echo "$cmd_name:$(date -I) - exporting conda environment to $export_path"
conda env export -n $env_name --from-history > $export_path
echo "$cmd_name:$(date -I) - copying environment file ${export_path}.tmp"
cp $export_path ${export_path}.tmp
echo "$cmd_name:$(date -I) - removing prefixed conda environment $export_path"
sed '$ d' ${export_path}.tmp > $export_path
echo "$cmd_name:$(date -I) - removing file ${export_path}.tmp"
rm -f ${export_path}.tmp
echo "$cmd_name:$(date -I) - finished conda environment to $export_path"