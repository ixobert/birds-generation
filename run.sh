set -e 0
echo $(pwd)
cat tasks-extract-latent.txt | parallel -j 2  --delay 10 {} 