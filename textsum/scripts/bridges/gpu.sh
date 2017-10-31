### for XSede comet cluster ###
### submit sbatch ---ignore-pbs train-2-gpu.sh
#SBATCH --ntasks-per-node 2
#SBATCH --time="24:00:00"
#SBATCH --gres=gpu:k80:1
#SBATCH -p GPU-shared
#SBATCH -N 1

echo $HOSTNAME
echo "Running on gpu"
python main.py --use_cuda\
    --vecdata /pylon5/ci560ip/bchen5/1001Nights/standard_giga/train/train_data_std_v50000.pkl\
    --model_fpat model/s2s-s%s-e%02d.model\
    --save_path /pylon5/ci560ip/bchen5/1001Nights/\
    --mode train
