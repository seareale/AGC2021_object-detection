find . -name ".ipynb_checkpoints" -type d -exec rm -r {} \; 
find . -name "__pycache__" -type d -exec rm -r {} \; 
find . -name ".vscode" -type d -exec rm -r {} \; 
find . -name ".lh" -type d -exec rm -r {} \; 
find . -name ".history" -type d -exec rm -r {} \;
find ./seareale -name "*.json" -type f -exec rm {} \;
find . -name "seareale.zip" -type f -exec rm {} \;

zip -r seareale.zip ./seareale
mv seareale.zip /ssd_2/RecycleTrash/org/mm_65/Downloads/제출/
