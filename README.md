# About the project
This project analysed the entire corpus (as of 2019) of National Occupational Standards (NOS) for Skills Development Scotland.

## Authors

Stef Garasto, Jyldyz Djumalieva

# Prerequisites
1. Python 3.6 and (ideally) Conda.
2. Homebrew and the cask plugin if on OSX.
3. The NOS dataset in pdf or json format. For this contact the Open Jobs Team or Skills Development Scotland. In theory, most (if not all) of the pdfs are available <a href="https://www.ukstandards.org.uk/">online</a>.

# Getting started

## Installation
1. <b> Clone the repo. </b> In a terminal, after navigating to the folder where you want to install the skill_demand repo, type:

<code>
git clone https://github.com/nestauk/openjobs-SDS-NOS-2019.git
</code>

2. <b> Create the conda environment. </b> The easiest way is the following. In a terminal, first navigate to the project folder. You are in the right folder, if it contains a file called <code> conda_environment.yaml</code>. Then type:

<code>conda env create -f conda_environment.yaml</code>

Note that the environment will be called "nos_analysis". If you want to give it a different name, then you need to make a copy of the file <code>conda_environment.yaml</code>, change the name in the first line of the new file and run the command above with the name of the new file. Making a copy means that it can be pushed to github without overriding the main file (better yet, if it's in a separate branch).

3. <b> Install Textract. </b>

See instructions <a href="https://textract.readthedocs.io/en/stable/installation.html">here</a>.

# Workflow
