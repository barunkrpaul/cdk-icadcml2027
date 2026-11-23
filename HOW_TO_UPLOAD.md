# Putting this repository on GitHub

1. On github.com create a new, empty, public repository. A suggested name is
   `cdk-icadcml2027`. Do not tick "Add a README".
2. Either drag the contents of this folder (not the folder itself) into the
   "uploading an existing file" page and commit, or use the command line:

       cd cdk-icadcml2027
       git init
       git add .
       git commit -m "CDK: code, results and camera-ready paper for ICADCML 2027"
       git branch -M main
       git remote add origin https://github.com/barunkrpaul/cdk-icadcml2027.git
       git push -u origin main

3. Copy the repository address into `\repourl` near the top of
   `paper/paper.tex` and of the response-to-reviewers source, then recompile.
4. Optional: on zenodo.org, link the GitHub account, switch the repository on
   and publish a release. Zenodo then gives a citable DOI.

You can delete this file after uploading.
