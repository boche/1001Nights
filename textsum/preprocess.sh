### for CMU rocks cluster ###
#PBS -j oe
#PBS -k oe
#PBS -o log
#PBS -d .
#PBS -V
#PBS -l walltime=48:00:00
#PBS -N Seq2Seq
#PBS -q default

## for XSede comet cluster ###
### submit sbatch ---ignore-pbs train-2-gpu.sh
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --output=log/slurm-%j.out
#SBATCH --export=ALL
#SBATCH --time="24:00:00"

module load gcc-4.9.2
echo $HOSTNAME
echo "Running on cpu"
python preprocess.py
#python preprocess.py --mode compress
