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
### Detection of near duplicates.

To identify near-duplicate NOS (that is, documents with a high amount of overlap - similar to plagiarism), we used a technique called <a href="https://en.wikipedia.org/wiki/Locality-sensitive_hashing">Locality Sensitive Hashing</a>. Specifically, we a) detected near duplicates and b) grouped them into non-overlapping groups (on the assumptions that such groups can be found exactly, and not that we have to find the best split into non-overlapping groups that approximate all the actual connections). A standalone version of the code used to do this can be found in the script <code>apply_LSH.ipynb</code>. Relevant functions have also been factored out in the script <code>lsh_func.py</code>.

It is worth noting that, originally, this code was developed to work with short texts. Here we used it on long texts, but because the intention since the beginning was to have a human reviewing the results. It might need to be assessed for accuracy and, if needed, improved appropriately, if used on long texts with no human in the loop.

### Predictive model for Education and Experience requirements.

Online job advert data from Burning Glass was used for this project. Specifically, we were interested in leveraging information on education and experience requirements. However, only a small minority of job adverts had this information, so we built a predictive model to fill the gaps.

When predicting educational requirements, we classified each job advert into "pregraduate / graduate / postgraduate" (that is, a coarser discretisation of years-based educational requirements) based on several attributes of the job description. These were: job title, "skills" needed, salary, SOC code, whether it's based in London or not (the latter was to adjust for the "London wage").

The steps taken were:
<ul>
<li>Phase 1. Within all the job adverts, check which SOC codes are matched to the same education category more than 90% of the time. This gives a group of “matched” SOC code. Classify a job advert based on the most likely category for its SOC codes, if its SOC code belongs to the group identified above. Unclassified job adverts move to the next stage.</li>
<li>Phase 2. For each unmatched SOC code, take its 10 most common job titles. For each job title, check if it’s matched to the same education category more than 90% of the time within the whole job adverts dataset. This gives a group of “matched” job titles. Classify a job advert based on the most likely category for its job titles, if its job titles belongs to the group identified above. Unclassified job adverts move to the next stage.</li>
<li>Phase 3. Train a random forest classifier to predict the educational category for all the other job adverts. Use all the features described above as input features.</li>
</ul>

The same steps were used to predict experience requirements, split into three levels: Junior, Middle and Senior.

The code with the model can be found in <code>NOS_pathways_and_skills/bg_build_hybrid_finalmodel_fullcv.py</code>. Note that it will not run out-of-the-box, because some inputs are likely missing. I could not include all inputs because we can not use Burning Glass data anymore. However, hopefully is a useful starting point.

In the folder <code>results/classifier_experience_education</code> there are some results from the model. For example, the file <code>finalmodel_occupations_matched_to_eduv2_category1_90_20190729.txt</code> shows which SOC codes were matched to which category of educational requirements and with what percentage. There are also txt files with the model performance compared to those of a 'dummy' classifier, as well as figures showing confusion matrices for the results.

The caveat is that, upon further reflection and with inputs from others, this approach would not be recommended, for ethical reasons. Specifically, estimating educational requirements for individual job adverts might solidify the "status quo". For example, since most data science jobs require a master at the moment then all data science jobs will be predicted to require a master, when it might/should not be the case.

### Finding exact occurrences of skills from pre-defined list within text.

For a more up-to-date way of doing this, see this <a href="https://data-analytic-nesta.slack.com/archives/CAYNV0XFZ/p1592820231035300">slack thread</a> (also <a href="https://docs.google.com/document/d/1nw9h_HzFknDcalXwrAzwF0SIlrNldZsl_w1zz0LNefg/edit?usp=sharing">here</a>).
