#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 15:02:13 2019

@author: stefgarasto

This script is to define the Skills Taxonomy class.
This class is used to load and process all iterations of Nesta's skills taxonomy.
Notes:
1. Due to restrictions on what we can publish, as it is the script won't work
with skills taxonomy 1.0 if cloned from Github. This is because we can not publish
the "bottom_layer" of the skills taxonomy 1.0.
"""

# Imports
from collections import Counter
from copy import deepcopy
import json
import pandas as pd
import pickle
import os
import requests
from utils_general import flattencolumns, aug_nesta_colours

#####
# General helper variables
#####

which_taxonomies = ['1_0','2_0']

# Only skills taxonomy 1 has assigned colours for now
# It's possible to modify the assignment of individual colours to skill clusters if desired
# assign cluster names to colours
first_level_colours = {'1_0': {'information technology':[93/255,196/255,200/255],
                       'education, sales and marketing': [217/255,49/255,131/255],
                       'engineering, construction and transport': [244/255,169/255,61/255],
                       'business administration': [236/255,93/255,42/255],
                       'health and social care': [189/255,215/255,69/255],
                       'science and research': [172/255,48/255,227/255],
                       'uncertain': [0/255,0/255,0/255]},
                       '2_0': {}}

# assign capitalized cluster names to colours
First_level_colours={}
for which_taxonomy in which_taxonomies:
    First_level_colours[which_taxonomy] = {}
    for t in first_level_colours.keys():
        First_level_colours[which_taxonomy][t.capitalize()]=first_level_colours[t]

all_clus_names = {'1_0':['legacy mainframe',
 'bi and data warehousing',
 'business analysis and it projects',
 'mainframe programming',
 'software development',
 'data engineering',
 'servers and middleware',
 'web development',
 'app development',
 'windows programming methods',
 'windows programming tools',
 'it security standards',
 'it security operations',
 'it security implementation',
 'intelligence',
 'networks',
 'system administration',
 'it support',
 'audit and compliance',
 'financial asset management',
 'securities trading',
 'insurance and lending',
 'accounting admin',
 'accounting and financial management',
 'payroll and tax accounting',
 'accounting software',
 'retail management',
 'hr management',
 'recruitment',
 'business management',
 'employee development',
 'premises security',
 'legal services',
 'office administration',
 'claims administration',
 'logistics administration',
 'supply chain management',
 'procurement',
 'shipping and warehouse operations',
 'biofuels',
 'health, safety and environment',
 'environmental planning',
 'aviation',
 'energy',
 'hydrology',
 'solar and wind energy',
 'construction engineering',
 'civil engineering',
 'oil and gas extraction',
 'structural engineering',
 'welding and machining',
 'heating, ventilation and plumbing',
 'driving and automotive maintenance',
 'electrical work',
 'construction',
 'design and process engineering',
 'manufacturing methods',
 'electrical engineering',
 'electronics',
 'automotive engineering',
 'infectious diseases',
 'pathology',
 'genomics and dna sequencing',
 'autoimmune conditions',
 'autoimmune cardiovascular disease',
 'pathophysiology',
 'cell biology',
 'molecular biology processes',
 'flow cytometry',
 'developmental biology',
 'molecular biology of cancer',
 'histology',
 'synthetic biology',
 'tomography and microscopy',
 'physics and math',
 'nanotechnology',
 'microfluidics',
 'teaching',
 'clinical research',
 'biomedical research',
 'research methods and statistics',
 'cell examination',
 'chromosome examination',
 'biochemistry',
 'biotechnology manufacturing',
 'laboratory techniques',
 'tissue culture',
 'prosthodontics and orthodontics',
 'dental assistance',
 'patient support',
 'endodontics and dental appliances',
 'dermatology',
 'ophthalmology',
 'surgical procedures',
 'reproductive health',
 'phlebotomy',
 'screening and immunisation',
 'general practice',
 'nutrition and diabetes management',
 'public health programmes',
 'mental health',
 'social work and caregiving',
 'physiotherapy and beauty',
 'neurological disorders',
 'speech and hearing therapy',
 'condition aneurysm',
 'critical care',
 'respiratory disease',
 'cardiology',
 'diagnostic imaging',
 'cardiac surgery',
 'medical coding',
 'medical admin',
 'pharmacy',
 'patient assistance and care',
 'clinical information systems',
 'nephrology',
 'gynecology and urology',
 'surgery',
 'oncology',
 'anesthesiology',
 'gastroenterology',
 'medical device sales',
 'marketing research',
 'advertising',
 'digital marketing',
 'marketing strategy and branding',
 'general sales',
 'complex sales',
 'retail',
 'media relations',
 'journalism and writing',
 'event planning',
 'web content management',
 'graphic and digital design',
 'animation',
 'digital content authoring',
 'multimedia production',
 'archiving and libraries',
 'extracurricular activities and childcare',
 'languages',
 'low vision support'],
    '2_0': []}

########
# Functions to process files containing the structure of a skills taxonomy
# Functions for both iterations of Nesta's skills taxonomy are included
########


def main_process_taxonomy_1(bg_clusters_dir, bg_clusters_file):
    """
    Read and parse files with the structure of the FIRST iteration of Nesta's skills taxonomy.
    This is the taxonomy created from clustering Burning Glass skills

    Inputs:
    bg_cluster_dir, bg_cluster_file: folder and filename of the file containing the
    bottom layer of the taxonomy. The bottom layer is a dictionary where each entry
    (key, value) is defined as follow. Key = name of a skill in the taxonomy.
    Value = cluster in the most granular level of the taxonomy to which that skill belongs.
    Note that bottom layer for Nesta's skills taxonomy 1.0 should NOT be made public
    (too many Burning Glass skills) to this function as it is should break.

    (Hidden) inputs:
    skills_taxonomy_json: location of the json file which contains the hierarchical
    structure of the taxonomy. This is the file that contains various per-cluster
    attributes, like average salary and growth.

    Outpus:
    - The bottom_layer dictionary.
    - bottom_clusters_names: a list of the skills in the taxonomy
    - skill_cluster_membership: a skills membership dictionary, where each entry
    (key, value) is defined as follow. Key = name of a cluster in the most granular
    level of the taxonomy. Value = list of skills in that cluster.
    - skills_taxonomy_full: a json containing the full, unprocessed structure of the
    taxonomy (with only Burning Glass skills that can be published).
    """
    try:
        with open(os.path.join(bg_clusters_dir,bg_clusters_file),'rb') as infile:
            bottom_layer = pickle.load(infile)
    except:
        print((f'The file {bg_clusters_dir}/{bg_clusters_file} needed for the '
            'skills taxonomy 1.0 does not exist'))

    # Collect skills in clusters
    skill_cluster_membership = {}
    for clus in Counter(bottom_layer.values()):
        cluster_skills = [elem for elem in bottom_layer if
                          bottom_layer[elem] == clus]
        if clus=='condition aneurysm':
            skill_cluster_membership['treatment of aneurysms'] = cluster_skills
        else:
            skill_cluster_membership[clus] = cluster_skills

    # load the full hierarchy of the taxonomy
    skills_taxonomy_json = '/Users/stefgarasto/Google Drive/Documents/data/skills-taxonomy/hierarchy8.json'
    with open(skills_taxonomy_json,'r') as f:
        skills_taxonomy_full = json.load(f)

    bottom_clusters_names = list(bottom_layer.keys())
    # Return the list of attributes
    return (bottom_layer, bottom_clusters_names,
        skill_cluster_membership, skills_taxonomy_full)


def process_taxonomy_1(bottom_layer, skills_taxonomy_full):
    """
    Further processing of the structure of the skills taxonomy 1.0 to get
    list of published skills, growth, average salary, etc

    Inputs:
    - bottom_layer dict. The bottom layer is a dictionary where each entry
    (key, value) is defined as follow. Key = name of a skill in the taxonomy.
    Value = cluster in the most granular level of the taxonomy to which that skill
    belongs.
    - skills_taxonomy_full: json-style dict describing the full taxonomy (with
    only Burning Glass skills that can be published).

    Outputs:
    - a dictionary with multiple entries, one for each attribute of the taxonomy.
    """
    # further processing
    def get_top5_skills(skills_dict, skills_var_name = 'top5_skills'):
        """" Specific instance of function get_named_attribute applied to the
        taxonomy attribute "top5_skills.
        The difference is that ultimately it returns a list of skills """
        list_of_skills = []
        if skills_var_name in skills_dict.keys():
            list_of_skills.append(skills_dict[skills_var_name])
            # if possible, go down a level
        if 'children' in skills_dict.keys():
            return list_of_skills + get_children_skills(skills_dict['children'],
                                                        skills_var_name)
        else:
            # exit recursion
            return list_of_skills

    def get_children_skills(list_of_dicts, skills_var_name = 'top5_skills'):
        """" Specific instance of function get_children_attr applied to the
        taxonomy attribute "top5_skills.
        The difference is that it returns a list of skills """
        # needs to return a skill
        list_of_skills = []#*len(list_of_dicts)
        for ix,skills_dict in enumerate(list_of_dicts):
            list_of_skills.append(get_top5_skills(skills_dict, skills_var_name))
        return flatten_lol(list_of_skills)

    def split_top_skills(list_of_skills):
        """ the top skills for each cluster are a list of tuples: (skill, skill "prevalence")
        This function transforms the list of tuples into a list of skills only"""
        list_of_skills = flatten_lol([[v[0] for v in t] for t in list_of_skills])
        return list_of_skills

    def split_top5_skills(list_of_skills):
        """ each top5 skill entry is a string of 5 skills separated by a comma.
        Split each string into the constituent skills """
        list_of_skills = flatten_lol([t.split(',') for t in list_of_skills])
        return [t.strip() for t in list_of_skills]

    def get_named_attribute(skills_dict, attr_name = 'mention_growth'):
        """ recursive function that collects all values of the attribute name for
        clusters at all level of the taxonomy. It starts from high level clusters
        and then visit all children clusters

        Inputs:
        - skills_dict: structure of the skills taxonomy 1.0 (as loaded from json file)
        - attr_name: attribute of interest

        Either goes into recursion or it outputs a list of tuples: each tuple has
        the cluster name and the attribute value"""
        attr_list = []
        if attr_name in skills_dict.keys():
            attr_list.append((skills_dict['name'], skills_dict[attr_name]))
            # if possible, go down a level (children clusters)
        if 'children' in skills_dict.keys():
            return attr_list + get_children_attr(skills_dict['children'],
                                                        attr_name)
        else:
            # exit recursion if there is no children cluster
            return attr_list

    def get_children_attr(list_of_dicts, attr_name = 'mention_growth'):
        """ functions that cycles through all the children clusters and
        applies the function "get_named_attribute"

        Inputs:
        -list of dicts: children clusters as a list of dictionaries
        -attr_name: attribute of interest

        The output is a list of tuples: each tuple has the cluster name and the
        attribute value
        """
        # needs to return a skill
        list_of_skills = []
        for ix,skills_dict in enumerate(list_of_dicts):
            list_of_skills.append(get_named_attribute(skills_dict, attr_name))
        return flatten_lol(list_of_skills)

    FLAG= True
    iter_number = 0
    # Collect all the public Burning Glass skills that live in the
    public_skills = get_top5_skills(skills_taxonomy_full)
    # split each string of 5 skills
    public_skills = list(set(split_top5_skills(public_skills[1:])))
    # assign to each public burning glass skill its low level cluster
    public_skills_clusters = {}
    for skill in public_skills:
        tmp = bottom_layer[skill]
        if tmp == 'condition aneurysm':
            tmp = 'treatment of aneurysms'
        public_skills_clusters[skill] = tmp

    # convert to Dataframe and group by cluster names
    public_skills_df =pd.DataFrame.from_dict(public_skills_clusters, orient='index').groupby(0)

    # recreate the dict containing low level clusters and their constituent skills
    # using only public Burning Glass skills
    public_skills_membership = {}
    for name, g in public_skills_df:
        public_skills_membership[name]= list(g.index)

    # Get all top skills per cluster [this is mostly for completeness in parsing
    # taxonomy hierarchy]
    public_skills_full = get_top5_skills(skills_taxonomy_full, skills_var_name = 'top_skills')
    public_skills_full = list(set(split_top_skills(public_skills_full[1:])))

    # Get the prevalence of each skill cluster (proportion of jobs)
    attr_list = get_named_attribute(skills_taxonomy_full, attr_name = 'prop_jobs')
    prop_jobs_dict = dict(zip([t[0] for t in attr_list],[t[1] for t in attr_list]))

    # Get the predicted growth for each skill cluster
    attr_list= get_named_attribute(skills_taxonomy_full, attr_name = 'mention_growth')
    growth_dict = dict(zip([t[0] for t in attr_list],[t[1] for t in attr_list]))

    # Get the average salary range for each skill cluster
    attr_list= get_named_attribute(skills_taxonomy_full, attr_name = 'avgsalary_range')
    avgsalary_dict = dict(zip([t[0] for t in attr_list],[t[1] for t in attr_list]))

    #% Get crosswalks from one taxonomy layer to another
    tax_first_layer = [t['name'] for t in skills_taxonomy_full['children']]
    tax_first_to_second = {}
    tax_second_to_first = {}
    tax_second_to_third = {}
    tax_third_to_second = {}
    for ix,layer1 in enumerate(tax_first_layer):
        level_down_from_first = skills_taxonomy_full['children'][ix]
        tax_first_to_second[layer1] = [t['name'] for t in level_down_from_first['children']]
        # second layer
        for ix2,layer2 in enumerate(tax_first_to_second[layer1]):
            tax_second_to_first[layer2] = layer1
            level_down_from_second =  level_down_from_first['children'][ix2]
            tax_second_to_third[layer2] = [t['name'] for t in level_down_from_second['children']]
            # third layer
            for layer3 in tax_second_to_third[layer2]:
                tax_third_to_second[layer3] = layer2

    #% I think this is legacy code from the SDS project. Can probably be deleted
    skills_matches = {'budgeting': 'budget planning',
                      'project management': 'project planning and development skills',
                      'wind turbine technology': 'wind turbines',
                      'calculation': 'calculator',
                      'writing': 'report writing',
                      'planning':'project planning and development skills',
                      'autocad': 'modelling software',
                      '3d autocad': '3d modelling software'}

    # assign all variables to the output dictionary
    return_dict = {}
    for var in ['public_skills_full', 'public_skills_membership', 'public_skills_clusters',
           'prop_jobs_dict', 'growth_dict', 'avgsalary_dict',
           'tax_first_layer', 'tax_first_to_second', 'tax_second_to_first',
           'tax_second_to_third', 'tax_third_to_second', 'skills_matches']:
           return_dict[var] = eval(var)
    return return_dict

def main_process_taxonomy_2(esco_clusters_dir,esco_clusters_file):
    """
    Read and parse files with the structure of the "skeleton" of the
    SECOND iteration of Nesta's skills taxonomy.
    This is the taxonomy whose "skeleton" is created from clustering ESCO skills.

    Inputs:
    esco_cluster_dir, esco_cluster_file: folder and filename of the file containing
    the Dataframe of all ESCO skills together with their cluster membership and
    some other attributes (like the description). One row = one ESCO skill.

    (Hidden) inputs:
    skills_taxonomy_json: location of the json file which contains the hierarchical
    structure of the taxonomy. This is the file that contains various per-cluster
    attributes, like average salary and growth.

    Outpus:
    - The bottom_layer dictionary. The bottom layer is a dictionary where each entry
    (key, value) is defined as follow. Key = name of a skill in the taxonomy.
    Value = cluster in the most granular level of the taxonomy to which that skill belongs.
    - bottom_clusters_names: a list of the skills in the taxonomy
    - skill_cluster_membership: a skills membership dictionary, where each entry
    (key, value) is defined as follow. Key = name of a cluster in the most granular
    level of the taxonomy. Value = list of skills in that cluster.
    - skills_taxonomy_full: the Dataframe containing the full structure of the
    taxonomy (only with those ESCO skills that belong to a cluster).
    - excluded_rows: a Dataframe containing those ESCO skills that do not belong
    to any cluster.

    Note for internal use: esco_clusters_dir is included for uniformity with the
    function for the skill taxonomy v1. Overall, some of the attributes defined
    by this function might be redundant or unnecessary. Might be good to clean it up.
    """
    print(f"Loading the taxonomy from {esco_clusters_file}")
    # Load the taxonomy structure in dataframe format (skills and clusters)
    esco_clusters_df = pd.read_csv(esco_clusters_file)
    # change the alternative labels column from string of text to a list of skills
    esco_clusters_df.alt_labels = esco_clusters_df.alt_labels.map(
        lambda x: x.split('\n') if isinstance(x,str) else [])

    # Only retain ESCO skills that have been clustered
    clustered_rows = esco_clusters_df.label_level_1.notna() #level_3>=0
    excluded_rows = esco_clusters_df[~clustered_rows]
    esco_clusters_df = esco_clusters_df[clustered_rows].reset_index(drop=True)

    # build the bottom layer dictionary
    bottom_layer = dict(zip(esco_clusters_df.preferred_label,
        esco_clusters_df.level_3))
    # build the skill_cluster_membership dictionary
    skill_cluster_membership = dict()
    for name,g in esco_clusters_df.groupby('level_3'):
        if name>=0:
            skill_cluster_membership[name] = list(g.preferred_label)

    # return list of attributes
    bottom_clusters_names = list(bottom_layer.keys())
    return (bottom_layer, bottom_clusters_names,
        skill_cluster_membership, esco_clusters_df, excluded_rows)


def process_taxonomy_2(bottom_layer,skills_taxonomy_full):
    '''
    Further processing of the structure of the skills taxonomy 2.0

    Inputs:
    - bottom_layer dict. The bottom layer is a dictionary where each entry
    (key, value) is defined as follow. Key = name of a skill in the taxonomy.
    Value = cluster in the most granular level of the taxonomy to which that skill
    belongs.
    - skills_taxonomy_full: DataFrame describing the full taxonomy. Each row is
    a [ESCO] skill. Columns needed are: "id" or "skill_id"= ID value for each skill;
    "alt_labels" = alternative labels (as a list of strings) for each skill.

    Outputs:
    - a dictionary with one entry ("alt_labels"): a pandas Series with all the
    alternative labels for the skills comprising the taxonomy. Each row in the
    Series is an alternative label (index) and the ID of the corresponding skill
    '''
    # Extract the relevant columns: for each row the skill ID and the alternative labels
    try:
        # use the column "id" if it's in the dataframe
        sub_df = skills_taxonomy_full.reset_index()[['id','alt_labels']]
        id_col = 'id'
    except:
        # if not, use the column "skill_id"
        sub_df = skills_taxonomy_full.reset_index()[['skill_id','alt_labels']]
        id_col = 'skill_id'
    # distribute the list of alternative labels per row across multiple columns
    # Some (row,columns) combination contain NaNs
    sub_df = flattencolumns(sub_df)
    # collect names of columns with alternative labels
    alt_columns = [t for t in sub_df.columns if 'alt_labels' in t]
    # unroll columns with alternative labels. This kind of goes from a wide
    # to a long dataframe structure with the same skill ID repeated on multiple
    # rows (as many times as there are alternative labels for that skill ID)
    # To do it quickly, it builds a list of pandas Series based on individual
    # columns, then it concatenates them
    df_list = []
    for col in alt_columns:
        df_list.append(sub_df[[col,id_col]].dropna().set_index(col))
    # return the unrolled Series of alternative labels
    return {'alt_labels': pd.concat(df_list)}

#%% define taxonomy class
class SkillTaxonomy:
    """
    SkillsTaxonomy class.

    This class should contain all that's needed to work with the Skills Taxonomy 2.0.
    It reads the csv file containing the structure of the taxonomy, parses it
    and stores relevant attributes.
    If needed, parsing can be expanded upon to include more attributes.

    Inputs needed when initialising the class:
    - name of the taxonomy. This can only be 1.0 or 2.0.

    Attributes of the class:
    - The name and location of the csv file containing the taxonomy structure.
    To avoid problems, these are fixed. These files should be provided with the Github repo
    - "Main_processing" and "further_processing": functions to parse the csv/json files above.
    - The bottom_layer dictionary. The bottom layer is a dictionary where each entry
    (key, value) is defined as follow. Key = name of a skill in the taxonomy.
    Value = cluster in the most granular level of the taxonomy to which that skill belongs.
    - skill_list: a list of the skills in the taxonomy
    - skill_cluster_membership: a skills membership dictionary, where each entry
    (key, value) is defined as follow. Key = name of a cluster in the most granular
    level of the taxonomy. Value = list of skills in that cluster.
    - skills_taxonomy_full: the Dataframe or json dict containing the full (public)
    structure of the taxonomy.
    - [for skills taxonomy 2.0 only] excluded_rows: a Dataframe containing those
    ESCO skills that do not belong to any cluster.
    - clus_names: a list of cluster names at the most granular level of the taxonomy

    TODOs (internal only):
    1. it's better to make the skill list a dataframe with unique IDs
    for each skills (for esco these should be related to the IDs in the
    original dataframe)
    2. Note that the class is organised this way because it needs to be usable
    with both taxonomies. Can be improved.
    """
    def __init__(self, taxonomy_name):

        self.taxonomy_name = taxonomy_name
        if taxonomy_name == '1_0':
            self.main_processing = main_process_taxonomy_1
            self.further_processing = process_taxonomy_1
            self.main_dir = '/Users/stefgarasto/Google Drive/Documents/scripts/NOS/'
            self.main_file = '/'.join([self.main_dir,
                'bottom_cluster_membership.pkl'])
        elif taxonomy_name == '2_0':
            self.main_dir = ('/Volumes/ssd_data/textkernel/data/aux')
            #'/Users/stefgarasto/Local-Data/scripts'
            #    '/openjobs-main/esco_map/data_processed/')
            self.main_file = os.path.join(self.main_dir,
             'ESCO_Essential_clusters_May2020_coreness.csv')
                #'skills_clusters_v03_2020_03_20.csv')
            self.main_processing = main_process_taxonomy_2
            self.further_processing = process_taxonomy_2
        else:
            assert(taxonomy_name in ['1_0','2_0'])

        # collect fundamental attributes of the skills taxonomy
        results = self.main_processing(
                self.main_dir,self.main_file)

        # unpack output into individual variables
        if len(results)==5:
            bottom_layer, skill_list, skill_cluster_membership, \
                skills_taxonomy_full, excluded_rows = results
        else:
            bottom_layer, skill_list, skill_cluster_membership, \
                skills_taxonomy_full = results
            excluded_rows = []

        # store attributes
        self.bottom_layer = bottom_layer
        self.skill_list = skill_list
        self.skill_cluster_membership = skill_cluster_membership
        self.skills_taxonomy_full = skills_taxonomy_full
        self.excluded_rows = excluded_rows
        self.clus_names = list(self.skill_cluster_membership.keys())

    def do_further_processing(self):
        ''' Further parsing of the taxonomy structures '''
        self.tax_attr = self.further_processing(self.bottom_layer,
            self.skills_taxonomy_full)

# create class for taxonomy 1.0
# Initialise and collect necessary attributes
try:
    taxonomy_1_0 = SkillTaxonomy('1_0')
    # add further attributes that might be useful
    taxonomy_1_0.do_further_processing()
except:
    taxonomy_1_0 = 'Unable to load Nesta skills taxonomy 1.0'
    print(taxonomy_1_0)

# create class for taxonomy 2.0
# Initialise and collect necessary attributes
taxonomy_2_0 = SkillTaxonomy('2_0')
# add further attributes that might be useful
taxonomy_2_0.do_further_processing()

###############
# Functions to load and use Emsi skills
###############
credentials_file = '/Users/stefgarasto/Local-Data/sensitive-data/emsi_api_credentials.csv'
credentials = pd.read_csv(credentials_file).T.to_dict()[0]

def obtain_oauth2():
    url = "https://auth.emsicloud.com/connect/token"

    payload = "client_id={}&client_secret={}&grant_type=client_credentials&scope={}".format(
    credentials['client_id'],
    credentials['secret'],
    credentials['scope']
    )
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.request("POST", url, data=payload, headers=headers)
    return response.json()

#oauth2 = obtain_oauth2()

def obtain_skills_list(oauth2=None):
    if not oauth2:
        oauth2 = obtain_oauth2()
    url = "https://skills.emsicloud.com/versions/latest/skills"
    headers = {'authorization': 'Bearer {}'.format(oauth2['access_token'])}
    response = requests.request("GET", url, headers=headers)
    return response.json()

def load_emsi_skills():
    emsi_skills = obtain_skills_list()
    # get the list
    emsi_skills = emsi_skills['skills']
    # remove certifications (they're likely to be USA specific anyway)
    emsi_skills = [t for t in emsi_skills if t['type']!='Certification']
    return emsi_skills

############
# Functions to load combined list of public skills that is locally stored
############
def load_all_public_skills():
    '''
    This function load all ESCO + ONET + EMSI + public BG skills
    '''
    # load ESCO and ONET
    with open('/Users/stefgarasto/Google Drive/Documents/data/ESCO/lookup_skills_esco_onet_bg.pkl' ,'rb') as f:
        skills_ext = pickle.load(f)
    skills_ext_long= [t for t in skills_ext if len(t)>3]

    #% # Load list of BG skills I can use for public output
    public_skills_full = taxonomy_1_0.tax_attr['public_skills_full']

    # Load EMSI skills
    skills_emsi = load_emsi_skills
    return skills_ext_long + public_skills_full + skills_emsi

def load_nesta_emsi_match():
    # THIS FUNCTION IS A LEGACY FUNCTION - it was created for the Skills Development
    # Scotland project in 2019
    # This is all that's needed to get the matches between Nesta and Emsi skills
    df_match_annotated = pd.read_csv(''.join(['/Users/stefgarasto/Google Drive/Documents/results/NOS/',
                     'nos_vs_skills/nesta_vs_emsi/nesta_to_emsi_bipartite_match_annotated.csv']))
    #df_match_annotated['good'][df_match_annotated['match']>94]='y'
    # add one row
    df_match_annotated = df_match_annotated.append(pd.DataFrame(
            data={'nesta':['supply chain'],'emsi': ['supply chain management'],
                  'match': [95],'good': ['y']}), sort=False)
    df_match_final = df_match_annotated[(df_match_annotated['match']>94) |
            (df_match_annotated['good']=='y')].reset_index()
    #'supply chain' --> 'supply chain management'
    return df_match_final
