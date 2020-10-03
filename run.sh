set -e 0
echo $(pwd)
/home/future/anaconda3/envs/scologan/bin/python ./src/train_vqvae.py +gpus=[0] debug=false dataset="taylor-fix" run_name="Init Train" tags=['vq-vae2', 'taylor_fix'] &\
sleep 3 &\
/home/future/anaconda3/envs/scologan/bin/python ./src/train_vqvae.py +gpus=[1] debug=false dataset="udem-birds" run_name="Init Train" tags=['vq-vae2', 'full_6_udem']
