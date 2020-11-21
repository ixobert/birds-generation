set -e 0
echo $(pwd)
cat tasks-train-vqvae2.txt | parallel -j 2  --delay 10 {} 