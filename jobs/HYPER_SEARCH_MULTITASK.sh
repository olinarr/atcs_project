#!/bin/bash
#SBATCH --job-name="olinarr_job"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=40:00:00
#SBATCH --partition=gpu_shared_course
#SBATCH --mail-type=END
#SBATCH --mail-user=olivieronardi@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --mem=60000M

module purge
module load pre2019
module load 2019

module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH

cp -r .data * $TMPDIR/
mkdir -p state_dicts

echo "finished moving! Starting with python..."

cd $TMPDIR/

for batch_size in 16 32 64 ; do
        for task_lr in 2e-5 2e-4 2e-3 ; do
            echo RUNNING HYPER PARAMETER SEARCH ON PARAMETERS: batch_size $batch_size task_lr $task_lr
            srun python3 -W ignore multitask.py --loss_print_rate 250 --epochs 4 --batch_size $batch_size --task_lr $task_lr
            mv ./state_dicts/* $HOME/project/state_dicts
        done
    done
done