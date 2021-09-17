#!/bin/bash
#SBATCH --job-name=aw
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --output=../complex_net_proposal/log/production-%j.qlog
#SBATCH -p standard
module load julia/1.5.3
module load gurobi/9.0.1

directory_structure="../complex_net_proposal/experiment_networks/"
input_paths=("jazz.net.clean.uel")

output_append="../complex_net_proposal/experiment_results/results_comp_find_issue.csv"

random_seed=20591

repititions=100

seed_method=$1


num_seeds=20
for base in ${input_paths[@]};
do
   full=$directory_structure
   full+=$base
   julia -O3 main_comp.jl $full $repititions $seed_method $num_seeds $random_seed $output_append
done

sed -r -i  "s/.*experiment_networks\///g" $output_append
sed -i 's/\.edges//g' $output_append
