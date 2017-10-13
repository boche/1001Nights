#!/bin/bash
### for CMU rocks cluster ###
#PBS -j oe
#PBS -k oe
#PBS -o log
#PBS -d .
#PBS -V
#PBS -l walltime=48:00:00
#PBS -N Seq2Seq
#PBS -q gpu

### for XSede comet cluster ###
### submit sbatch ---ignore-pbs train-2-gpu.sh
#SBATCH --ntasks-per-node 2
#SBATCH --time="24:00:00"
#SBATCH --gres=gpu:k80:1
#SBATCH -p GPU-shared
#SBATCH -N 1

#module load gcc-4.9.2
#module load cuda-8.0
#export CUDA_VISIBLE_DEVICES=`qstat -n $PBS_JOBID|awk 'END {split ($NF, a, "/"); printf ("%s\n", a[2])}'`
#echo $HOSTNAME
#echo "Running on gpu"
#echo "Device = $CUDA_VISIBLE_DEVICES"
#python main.py --build_vocab
#python main.py --build_emb
python main.py --use_cuda
