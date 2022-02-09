for i in {31..34}
    do
    cp SUBX_SG_gcf.sh SUBX_SG_gcf_tmp_${i}.sh

    echo "Running SUBX GCF $i"

    sed -i "s/NUM/$i/" SUBX_SG_gcf_tmp_${i}.sh

    sbatch SUBX_SG_gcf_tmp_${i}.sh

    rm SUBX_SG_gcf_tmp_${i}.sh
    done
