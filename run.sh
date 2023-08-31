set -e 0
echo $(pwd)
cat tasks-train-classifier.txt | parallel -j 6  --delay 10 {} 