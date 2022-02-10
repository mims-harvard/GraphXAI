for i in {42..44}
    do
    cp SUBX_SG.sh SUBX_SG_tmp_${i}.sh

    echo "Running SUBX $i"

    sed -i "s/NUM/$i/" SUBX_SG_tmp_${i}.sh

    sbatch SUBX_SG_tmp_${i}.sh

    rm SUBX_SG_tmp_${i}.sh
    done
