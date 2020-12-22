#!/bin/bash
#SBATCH --job-name=aw
#SBATCH --nodes=1
#SBATCH --mincpus=16
#SBATCH --time=94:00:00
#SBATCH --output=../complex_net_proposal/log/production-%j.qlog
#SBATCH -p bii
module load julia/1.5.3
module load gurobi/9.0.1

directory_structure="../complex_net_proposal/experiment_networks/"
input_paths=("${2}")
output_append="../complex_net_proposal/experiment_results/results_mcich_smc.csv"

random_seed=20591

repititions=100

seed_method=$1

blocking_method="MCICH_SMC"

num_seeds=20
for base in ${input_paths[@]};
do
   full=$directory_structure
   full+=$base
   julia -O3 main.jl $full $repititions $seed_method $num_seeds $random_seed $output_append $blocking_method
done

sed -r -i  "s/.*experiment_networks\///g" $output_append
sed -i 's/\.edges//g' $output_append
