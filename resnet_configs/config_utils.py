import os 
def generate_lr_momentum_config():
  momentum = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
  lr = [0.05, 0.1, 0.2, 0.4, 0.8]
  name_list = []
  with open('lr_momentum.yaml', 'w') as outfile:
    for item1 in lr:
      for item2 in momentum: 
        name = f'lr{str(item1).split(".")[1]}mom{str(item2).split(".")[1]}'
        name_list.append(name)
        config = '''
{0}:
  avg_pool_kernel_size: 100
  conv_kernel_sizes: [3,3,3,3] 
  num_blocks: [2,1,1,1] 
  num_channels: 64
  shortcut_kernel_sizes: [1,1,1,1] 
  drop: 0 # proportion for dropout 
  squeeze_and_excitation: 0 # True=1, False=0 
  max_epochs: 200
  optim: "sgd" 
  lr_sched: "CosineAnnealingLR"
  momentum: {1}
  lr: {2} 
  weight_decay: 0.0005 
  batch_size: 64
  num_workers: 2
  resume_ckpt: 0 # 0 if not resuming, else path to checkpoint  
  data_augmentation: 1 # True=1, False=0 
  data_normalize: 1 # True=1, False=0 
  grad_clip: 0.1\n'''.format(name, item2, item1)
        outfile.write(config)
  return name_list

def generate_batch_file(config_file_name, sh_file_name, name_list):
  sh_path = os.path.join('..', 'scripts', sh_file_name)
  shell_file = open(sh_path, 'w')
  for name in name_list:
    file_name = name + '.sbatch'
    shell_file.write('sbatch ' + file_name + '\n')
    path = os.path.join('..', 'scripts', file_name)
    with open(path, 'w') as outfile:
      script = '''
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=hpc4473C0
#SBATCH --output=outputs/{0}.out

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate harienv

python ../main.py --config ../resnet_configs/{1} --resnet_architecture {0}
'''.format(name, config_file_name)
      outfile.write(script.strip())
  shell_file.close()


if __name__ == "__main__":
  name_list = generate_lr_momentum_config()        
  generate_batch_file('lr_momentum.yaml', 'lr_momentum.sh', name_list)