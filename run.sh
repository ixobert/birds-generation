set -e 0
echo $(pwd)
cat tasks.txt | parallel -j 2  --delay 5 {} 