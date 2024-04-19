# Appendix for modelcomp

[modelcomp](https://github.com/gilramot/modelcomp) is a package developed for a microbiome research. This repo contains the files essential for reproducing the results of the research.

## Files

*base-article* - the base article for the research (uses its dataset)

*data* - the input dataset (*abundance.csv* stores the microbiomial data, *meta.csv* stores the target variable)

### results
A folder that contains subfolders corresponding to each version of the code. Within each version's subfolder, there are:
*main.py* - a file that can be used for replicate the results of the research, optimized for each version individually
*export* - the output of main.py

*disease-keys.txt* - the possible keys of the target variable and their values