#!/bin/bash
#SBATCH --job-name=aw
#SBATCH --nodes=1
#SBATCH --mincpus=12
#SBATCH --time=72:00:00
#SBATCH --output=../complex_net_proposal/log/production-%j.qlog
#SBATCH -p bii
module load julia/1.5.0

directory_structure="../complex_net_proposal/experiment_networks/"
input_paths=("astroph.edges" "wiki.edges" "fb-pages-politician.edges" "primary-school-proximity.edges" "slashdot0811.edges")

output_append="../complex_net_proposal/experiment_results/results_run_mcich.csv"

random_seed=20591

repititions=50

seed_method="random_k_core"

num_seeds=20
for base in ${input_paths[@]};
do
   full=$directory_structure
   full+=$base
   julia main.jl $full $repititions $seed_method $num_seeds $random_seed $output_append
done

sed -r -i  "s/.*experiment_networks\///g" $output_append
sed -i 's/\.edges//g' $output_append