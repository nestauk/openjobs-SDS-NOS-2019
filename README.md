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

The data consists of the dataset of <a href= "https://www.ukstandards.org.uk/">National Occupational Standards</a> (NOS). A snapshot of the dataset as it was in June 2019 is available on this <a href="https://s3.console.aws.amazon.com/s3/buckets/open-jobs-lake/">S3 bucket</a> both in the original pdf format (folder 'national-occupational-standards-pdf') and in the extracted json format (folder 'national-occupational-standards-json/extracted-json'). A subset of NOS (the more recently developed ones) is also available in a cleaner json format (folder 'national-occupational-standards-json/new-json'). In theory, most (if not all) of the pdfs are also available <a href="https://www.ukstandards.org.uk/">online</a> (which might be useful to check for updates), but there is no API access that I know of.

# Research overview
The project was split into three parts.

## Top level analysis and level of duplication
The first part aimed at mapping commonalities among NOS and at measuring the scale of overlap/duplication within the current NOS database. Together with a descriptive analysis of NOS characteristics, we used NLP to:
<ul>
<il> Identify standards that are near-duplicates and measure the overall level of overlap/duplication in the NOS database;</it>
<it> Derive keywords associated with individual standards and suites in a data-driven way; </il>
<il> Group standards based on the similarity of their keywords. </il>
</ul>

For more information, see the <a href="https://drive.google.com/file/d/12cFDX6XpujaotMZqKrObRKy1dmXmzQfp/view?usp=sharing">final report</a>.

## Analysis of progression pathways
The aim of this research project was to use NLP and Machine Learning to automatically organise groups of national occupational standards into progression pathways, which is currently a lengthy manual process. While there are many ways to define progression, here we assume that higher levels of progression are associated with more work experience, higher educational qualifications and greater earnings. Intuitively, more senior jobs and tasks often require advanced work experience and qualifications and offer higher levels of salary.

For more information, see the <a href="https://drive.google.com/file/d/1rKNA2hIxNPBK1GzWazLzVXYx4UVDZ-BG/view?usp=sharing">final report</a>.

## Analysis of skill content of NOS
The aim of this work package was to illustrate how skills could be automatically extracted from NOS descriptions using an extensive library of skills collected from multiple sources. We also evaluated the skill content of NOS to detect gaps in the coverage of skill domains. Potential gaps were analysed in terms of demand, market value and growth in demand for skills in the corresponding domains. These labour market indicators were derived from the first iteration of Nesta’s taxonomy of skills, built using online job adverts.

For more information, see the <a href="https://drive.google.com/file/d/1HMO0NAeDtZA41rYK-loOii8N-eT45JMg/view?usp=sharing">final report</a>.

## Relevant information for DfE project
1. Detection of near duplicates. TODO

Add where to find script and what to improve. Needs updating. Specifically, it was created to work with short texts originally. I used it on long texts, but because the intention since the beginning was to have a human reviewing the results. It might need to be assessed for accuracy and, if needed, changed if used on long texts with no human in the loop

2. Predictive model for Education and Experience requirements. TODO

ADD description from email. Add where to find some scripts, results and how to read them. Add ethical caveats and why is not recommended.
Specifically, estimating educational requirement for individual job adverts might solidify the "status quo". For example, since most data science jobs require a master at the moment then all data science jobs will be predicted to require a master, when it might/should not be the case

3. Finding exact occurrences of skills from pre-defined list within text.

For a more up-to-date way of doing this, see this <a href="https://data-analytic-nesta.slack.com/archives/CAYNV0XFZ/p1592820231035300">slack thread</a> (also <a href="https://docs.google.com/document/d/1nw9h_HzFknDcalXwrAzwF0SIlrNldZsl_w1zz0LNefg/edit?usp=sharing">here</a>).
