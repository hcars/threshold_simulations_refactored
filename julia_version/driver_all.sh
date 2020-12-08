input_paths=("enron.giant.clean.uel" "astroph.edges" "wiki.edges" "fb-pages-politician.edges" "slashdot0811.edges")


for base in ${input_paths[@]};
do
    sbatch driver_jl_mcich_smc_v01.sh random_k_core $base	
done
