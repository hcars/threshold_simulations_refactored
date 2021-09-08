#!/bin/bash
#SBATCH --job-name=aw
#SBATCH --nodes=1
#SBATCH --mincpus=16
#SBATCH --time=94:00:00
#SBATCH --output=./complex_nets_2021/log/production-%j.qlog
#SBATCH -p bii
module load julia/1.5.3
module load gurobi/9.0.1

directory_structure="../complex_net_proposal/experiment_networks/"
input_paths=""
output_append="./complex_nets_2021/experiment_results/results_mcich_smc_wiki_modified.csv"

random_seed=20591

repititions=10

seed_method="centola"

blocking_method="MCICH_SMC"

num_seeds=100
for base in ${input_paths[@]};
do
   if [ "$base" == "astroph.edges" ]
	then
	num_seeds=100
   elif [ "$base" == "fb-pages-politician.edges" ]
	then
	num_seeds=100
   elif [ "$base" == "wiki.edges" ]
	then
	num_seeds=100
	fi

   full=$directory_structure
   full+=$base
   julia -O3 main_wiki_mod.jl $full $repititions $seed_method $num_seeds $random_seed $output_append $blocking_method
done

sed -r -i  "s/.*experiment_networks\///g" $output_append
sed -i 's/\.edges//g' $output_append
