#!bin/bash

for i in {0..1}
    do
    cp SUBX_SG.sh SUBX_SG_tmp_$i.sh

    echo "Running SUBX $i"

    sed -i "s/NUM/$i"

    sbatch SUBX_SG_tmp_$i.sh

    rm SUBX_SG_tmp_$i.sh
    done