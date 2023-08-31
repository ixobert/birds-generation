set -e 0
echo $(pwd)
cat tasks-train-classifier.txt | parallel -j 4  --delay 10 {} 