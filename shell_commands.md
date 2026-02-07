# Push data to scratch remote
  $DVC push

  # Then commit the updated .dvc files to git
  git add data/cache/*.dvc data/cache/.gitignore
  git commit -m "Update DVC-tracked dataset caches"
  git push