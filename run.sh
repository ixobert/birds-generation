set -e 0
echo $(pwd)
cat tasks-train-classifier.txt | parallel -j 2  --delay 10 {} 