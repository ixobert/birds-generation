set -e 0
echo $(pwd)
/home/future/anaconda3/envs/scologan/bin/python ./src/train_vqvae.py +gpus=[0] debug=false tags=['debug']