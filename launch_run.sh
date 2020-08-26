CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo Current branch: $CURRENT_BRANCH
echo Run message: $1
git commit -am $1
git checkout runs
git merge $CURRENT_BRANCH
git push
git checkout $CURRENT_BRANCH