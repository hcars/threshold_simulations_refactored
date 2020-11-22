seeding_methods=("centola" "random_k_core")
for seeding_method in ${seeding_methods[@]}
do
        ./driver_jl_mcich_smc_v01.sh $seeding_method
        ./driver_jl_mcich_ilp_v01.sh $seeding_method
done