#!/usr/bin bash
datanames=('fewrel' 'fewrel2.0' 'cpr')
for dataname in ${datanames[@]}
do
    python run_discover.py \
        --dataname $dataname 
done