#!/bin/sh

# Generate directory structure without hidden files and exclude certain directories
echo '```' > TREE.md
tree -I '.git|node_modules|env|R|data|logs|stimuli' >> TREE.md
echo '```' >> TREE.md

# Sync TREE.md to README.md
awk '/<!-- TREE_START -->/{flag=1; next} /<!-- TREE_END -->/{flag=0} !flag' README.md > temp.md
echo "<!-- TREE_START -->" >> temp.md
cat TREE.md >> temp.md
echo "<!-- TREE_END -->" >> temp.md
mv temp.md README.md
