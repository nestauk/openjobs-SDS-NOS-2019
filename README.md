# About the project
This project analysed the entire corpus (as of 2019) of National Occupational Standards (NOS) for Skills Development Scotland.

## Authors

Stef Garasto, Jyldyz Djumalieva

# Prerequisites
1. Python 3.6 and (ideally) Conda.
2. Homebrew and the cask plugin if on OSX.

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

4. <b> Download the data. </b>

The data consists of the dataset of <a href= "https://www.ukstandards.org.uk/">National Occupational Standards</a> (NOS). A snapshot of the dataset as it was in June 2019 is available on this <a href="https://s3.console.aws.amazon.com/s3/buckets/open-jobs-lake/">S3 bucket</a> both in the original pdf format and in the extracted json format. A subset of NOS (the more recently developed ones) is also available in a cleaner json format. In theory, most (if not all) of the pdfs are also available <a href="https://www.ukstandards.org.uk/">online</a> (which might be useful to check for updates), but there is no API access that I know of.

# Workflow
The project was split into three parts.

## Top level analysis and level of duplication.
