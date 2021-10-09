#!/usr/bin/env bash
python3 -m mkdocs build
git checkout -b gh-pages-temp
git add -f site
git commit -m "gh-pages update"
git subtree split --prefix site -b gh-pages
git push -f origin gh-pages:gh-pages
git branch -D gh-pages
git checkout main
git branch -D gh-pages-temp
