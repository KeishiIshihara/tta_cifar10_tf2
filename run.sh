#!/bin/bash

start_time=`date +%s`

CONFIG=('01_no_aug_no_tta.yaml' '02_aug_no_tta.yaml' '03_no_aug_tta.yaml' '04_aug_tta.yaml')
SEED='46' # 42, 43, 44, 45, 46
RUN_ID=$(python utils/generate_runid.py)

echo '-------------------------'
echo "RUN_ID: $RUN_ID"
echo '-------------------------'

export RUN_ID

for conf in ${CONFIG[@]}; do
    echo
    command="python train.py --config $conf --seed $SEED"
    echo "$ $command"
    eval $command
done

cd utils
command="python plot_reports.py --filename tta_results_seed-${SEED}_runid-${RUN_ID}.png"
echo
echo $command
eval $command
cd ..

end_time=`date +%s`
run_time=$((end_time - start_time))
echo ${run_time}s
echo "scale=3; $run_time/60" | bc # scaleで少数第何位まで出力するか指定可能
