set -e 0
echo $(pwd)
/home/future/anaconda3/envs/scologan/bin/python ./src/train_vqvae.py +gpus=[1] debug=false tags=['debug']