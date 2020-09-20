CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT_MESSAGE=${@}
echo Current branch: $CURRENT_BRANCH
echo Run message: $COMMIT_MESSAGE
git commit -am "$COMMIT_MESSAGE"
git checkout runs
git merge $CURRENT_BRANCH
git push
git checkout $CURRENT_BRANCH