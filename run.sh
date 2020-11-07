set -e 0
echo $(pwd)
cat tasks-train-vqvae.txt | parallel -j 2  --delay 5 {} 