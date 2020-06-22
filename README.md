# Understanding the Role of Background Knowledge in Predictions; Ecotoxicological Effect Prediction. Master's thesis by Nils Petter Opsahl Skrindebakke.


# Install
conda create --name myenv --file spec-file.txt

# Directory Structure
TCE: Triple Contex Embedding in Python (under development).

data: Contains the training data.

graph_analysis: Tools for analysis of the KG.

kg: Contains the original KGs.

kg/processed: The results from the crawls.

obj: Contains a binary file used for cache when mapping to CID.

plots: Tools for plotting and printing the results.

prep: Tools for downloading and and prossess KGs, test and train data.

results: Binary file containing a dictionary with the results from the runs.

src: Contains the ML-models and other related tools to run and evaluate the predictions.

src/crawl: Contains the crawling and scoring algorithms.
