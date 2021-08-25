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
input_paths=("fb-pages-politician.edges")
output_append="./complex_nets_2021/experiment_results/k_core_results_epi_curves.csv"

random_seed=20591

repititions=20

seed_method="random_k_core"


num_seeds=30
for base in ${input_paths[@]};
do
   full=$directory_structure
   full+=$base
   julia -O3 main_epi_curve.jl $full $repititions $seed_method $num_seeds $random_seed $output_append
done

