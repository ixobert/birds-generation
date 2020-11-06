set -e 0
echo $(pwd)
cat tasks-train-prior.txt | parallel -j 2  --delay 5 {} 