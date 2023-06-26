set -e
#Get current branch name
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT_MESSAGE=$1


#Commit current changes
#if branch is runs, Please commit to another branch, then exit this script
if [ "$CURRENT_BRANCH" = "runs" ] ; then
    echo "Please commit to another branch."
    exit 1
fi

echo Current branch: $CURRENT_BRANCH
echo Run message: $COMMIT_MESSAGE
git commit -am "$COMMIT_MESSAGE"

# #Get freshly commit hash
# GIT_HASH=$(git rev-parse --short HEAD)
# echo "$GIT_HASH" > ./src/git_hash.txt
# git add ./src/git_hash.txt
# git commit ./src/git_hash.txt -m "add githash"

#Checkout to runs, merge with CURRENT_BRANCH and push
git checkout runs
git merge $CURRENT_BRANCH
git push
git checkout $CURRENT_BRANCH