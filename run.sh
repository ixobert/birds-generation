set -e 0
echo $(pwd)
/home/future/anaconda3/envs/scologan/bin/python ./src/train_vqvae.py nb_epochs=10 +gpus=[0]