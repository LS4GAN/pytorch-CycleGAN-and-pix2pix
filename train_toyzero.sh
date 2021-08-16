
#!/bin/bash

base_path="/sdcc/u/yhuang2/PROJs/GAN/pytorch-CycleGAN-and-pix2pix/"
config_path="${base_path}/configs"
script="${base_path}/train_toyzero.py"
runfiles_path="${base_path}/runfiles"

config_files=('config_toyzero_128x128_32_64.yaml' 'config_toyzero_512x512_8_16.yaml' 'config_toyzero_512x512_4_32.yaml' 'config_toyzero_512x512_2_64.yaml')
side=(128 512 512 512)
bsz=(32 8 4 2)
ngf=(64 16 32 64)
hours=(2 6 13 35)

for ((idx=0; idx<${#config_files[@]}; ++idx)); 
do
	config_file="${config_path}/${config_files[idx]}"
	job_name="${side[idx]}_${bsz[idx]}_${ngf[idx]}"
	time="${hours[idx]}:00:00"

	echo "$idx" "$config_file" "$jobname" "$time"

	# command

	sbatch_fname="${runfiles_path}/${job_name}.sbc"
	echo "#!/bin/bash" > $sbatch_fname
	echo "#SBATCH --partition=volta" >> $sbatch_fname
	echo "#SBATCH --account mlg-core" >> $sbatch_fname
	echo "#SBATCH --nodes=1" >> $sbatch_fname
	echo "#SBATCH --ntasks=1" >> $sbatch_fname
	echo "#SBATCH --gres=gpu:1" >> $sbatch_fname
	
	echo "#SBATCH --time=${time}" >> $sbatch_fname
	echo "#SBATCH --job-name=${job_name}" >> $sbatch_fname
	echo "#SBATCH --output=${runfiles_path}/${job_name}.out" >> $sbatch_fname
	echo "#SBATCH --error=${runfiles_path}/${job_name}.err" >> $sbatch_fname
	
	cmd="srun --unbuffered python ${script} ${config_file}"
	echo "${cmd}" >> "${sbatch_fname}"
	
	sbatch $sbatch_fname
done





####################################### historical #######################################
# srun --unbuffered python train.py --dataroot ~/PROJs/GAN/datasets/ls4gan/toyzero_cropped/toyzero_2021-06-29_safi_U --name toyzero --dataset_mode toyzero --input_nc 1 --output_nc 1 --model cycle_gan --max_dataset_size 1000 --batch_size 32
