#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 15:02:13 2019

@author: stefgarasto
"""

# ### Find average vector per skill cluster
# 

from utils_nlp import *
import pickle
from collections import Counter

first_level_colours = {'information technology':[93/255,196/255,200/255],
                       'education, sales and marketing': [217/255,49/255,131/255],
                       'engineering, construction and transport': [244/255,169/255,61/255],
                       'business administration': [236/255,93/255,42/255],
                       'health and social care': [189/255,215/255,69/255],
                       'science and research': [172/255,48/255,227/255],
                       'uncertain': [0/255,0/255,0/255]}

First_level_colours={}

for t in first_level_colours.keys():
    First_level_colours[t.capitalize()]=first_level_colours[t]
    
# In[83]:

all_clus_names = ['legacy mainframe',
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
 'low vision support']


clus_names_for_engineering = [
 'software development',
 'data engineering',
 'it support',
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
 'research methods and statistics',
 'biotechnology manufacturing',
 'employee development']
# 'business management',
# 'animation']
#  'audit and compliance',
# 'supply chain management',
# 'shipping and warehouse operations',

#%%
#Load the file with the skills taxonomy lower layer
with open(''.join(['/Users/stefgarasto/Google Drive/Documents/scripts/NOS/',
                       'bottom_cluster_membership.pkl']), 
              'rb') as infile:
    bottom_layer = pickle.load(infile)

nesta_skills = list(bottom_layer.keys())

with open('/Users/stefgarasto/Google Drive/Documents/data/ESCO/lookup_skills_esco_onet_bg.pkl' ,'rb') as f:
    skills_ext = pickle.load(f)
skills_ext_long= [t for t in skills_ext if len(t)>3]

# This is all that's needed to get the skills match between Nesta and Emsi
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

#%%
def load_and_process_clusters(model, ENG = False): 
    
    # Collect skills in clusters
    skill_cluster_membership = {}
    for clus in Counter(bottom_layer.values()):
        cluster_skills = [elem for elem in bottom_layer if                       
                          bottom_layer[elem] == clus]
        if clus=='condition aneurysm':
            skill_cluster_membership['treatment of aneurysms'] = cluster_skills
        else:
            skill_cluster_membership[clus] = cluster_skills
    
    print(list(skill_cluster_membership.keys())[::10])
        
    
    if ENG:
        skill_cluster_membership2 = {}
        for clus in skill_cluster_membership.keys():
            if clus in clus_names_for_engineering:
                skill_cluster_membership2[clus] = skill_cluster_membership[clus]
        skill_cluster_membership = skill_cluster_membership2
    
    # Generate lookup vecs using pre-trained GloVe model
    # I guess this is to get a mean vector for each skills cluster
    skill_cluster_vecs = {}
    full_skill_cluster_vecs = {}
    for clus in skill_cluster_membership:
        # get all bottom layer skills in cluster
        cluster_skills = skill_cluster_membership[clus]
        ## convert to underscore
        #new_skills = [convert_to_undersc(elem) for elem in cluster_skills]
        ## get skills with more than one term
        #other_skills = [elem.split() for elem in cluster_skills if len(elem)>1]
        #flat_other_skills = [item for sublist in other_skills for item in sublist]
        ## add skills with underscore and split skills
        #all_skills = new_skills + list(set(flat_other_skills))
        ## only keep those in the model
        #skills_in = [elem for elem in all_skills if elem in model]
        #if np.random.randn(1)>3.2:
        #    print(clus, len(cluster_skills), len(skills_in))
        skill_cluster_vecs[clus], full_skill_cluster_vecs[clus] = get_mean_vec(
                                                        cluster_skills, model)
    
    # check all skill clusters have a vector: check has to be empty
    check = [k for k,v in skill_cluster_vecs.items() if len(v.shape) == 0]
    assert(len(check)==0)
    #print('This should be empty.',check)
     
    #with open(os.path.join(output_dir, 'skill_cluster_vecs_pretrained.pkl'), 'wb') as f:
    #    pickle.dump(skill_cluster_vecs, f)
     
    # show words in the model that are closest to average vector for each skill cluster
    print('Showing representative words in the model for some skills cluster')
    for clus in list(skill_cluster_vecs.keys())[10:]:
        print(clus)
        print(model.similar_by_vector(skill_cluster_vecs[clus]))
        print('***********')
    print()

    # arrange all mean skill vectors in a matrix
    comparison_vecs = np.vstack(list(skill_cluster_vecs.values()))
    clus_names = list(skill_cluster_vecs.keys())
    
    #print(clus_names[:10], comparison_vecs.shape)
    return clus_names, comparison_vecs, (skill_cluster_vecs, full_skill_cluster_vecs)


#%%
# Load list of skills I can use for public output
import json
from copy import deepcopy

skills_taxonomy_json = '/Users/stefgarasto/Google Drive/Documents/data/skills-taxonomy/hierarchy8.json'
with open(skills_taxonomy_json,'r') as f:
    skills_taxonomy_full = json.load(f)

#%%
def get_top5_skills(skills_dict, skills_var_name = 'top5_skills'):
    list_of_skills = []
    if skills_var_name in skills_dict.keys():
        list_of_skills.append(skills_dict[skills_var_name])
        # if possible, go down a level
    if 'children' in skills_dict.keys():
        return list_of_skills + get_children_skills(skills_dict['children'], 
                                                    skills_var_name)
    else:
        # interrupt recursion
        return list_of_skills

def get_children_skills(list_of_dicts, skills_var_name = 'top5_skills'):
    # needs to return a skill
    list_of_skills = []#*len(list_of_dicts)
    for ix,skills_dict in enumerate(list_of_dicts):
        list_of_skills.append(get_top5_skills(skills_dict, skills_var_name))
    return flatten_lol(list_of_skills)
 
def split_top_skills(list_of_skills):
       list_of_skills = flatten_lol([[v[0] for v in t] for t in list_of_skills])
       return list_of_skills

def split_top5_skills(list_of_skills):
       list_of_skills = flatten_lol([t.split(',') for t in list_of_skills])
       return [t.strip() for t in list_of_skills]

def get_named_attribute(skills_dict, attr_name = 'mention_growth'):
    attr_list = []
    if attr_name in skills_dict.keys():
        attr_list.append((skills_dict['name'], skills_dict[attr_name]))
        # if possible, go down a level
    if 'children' in skills_dict.keys():
        return attr_list + get_children_attr(skills_dict['children'], 
                                                    attr_name)
    else:
        # exit recursion
        return attr_list
    
def get_children_attr(list_of_dicts, attr_name = 'mention_growth'):
    # needs to return a skill
    list_of_skills = []
    for ix,skills_dict in enumerate(list_of_dicts):
        list_of_skills.append(get_named_attribute(skills_dict, attr_name))
    return flatten_lol(list_of_skills)

#%% 
FLAG= True
iter_number = 0
public_skills = get_top5_skills(skills_taxonomy_full)
public_skills = list(set(split_top5_skills(public_skills[1:])))
public_skills_clusters = {}
for skill in public_skills:
    tmp = bottom_layer[skill]
    if tmp == 'condition aneurysm':
        tmp = 'treatment of aneurysms'
    public_skills_clusters[skill] = tmp
tmp =pd.DataFrame.from_dict(public_skills_clusters,orient='index').groupby(0)

public_skills_membership = {}
for name, g in tmp:
    public_skills_membership[name]= list(g.index)

public_skills_full = get_top5_skills(skills_taxonomy_full, skills_var_name = 'top_skills')
public_skills_full = list(set(split_top_skills(public_skills_full[1:])))

attr_list = get_named_attribute(skills_taxonomy_full, attr_name = 'prop_jobs')
prop_jobs_dict = dict(zip([t[0] for t in attr_list],[t[1] for t in attr_list]))

attr_list= get_named_attribute(skills_taxonomy_full, attr_name = 'mention_growth')
growth_dict = dict(zip([t[0] for t in attr_list],[t[1] for t in attr_list]))

attr_list= get_named_attribute(skills_taxonomy_full, attr_name = 'avgsalary_range')
avgsalary_dict = dict(zip([t[0] for t in attr_list],[t[1] for t in attr_list]))

#%%
tax_first_layer = [t['name'] for t in skills_taxonomy_full['children']]
tax_first_to_second = {}
tax_second_to_first = {}
tax_second_to_third = {}
tax_third_to_second = {}
for ix,layer1 in enumerate(tax_first_layer):
    tmp = skills_taxonomy_full['children'][ix]
    tax_first_to_second[layer1] = [t['name'] for t in tmp['children']]
    # second layer
    for ix2,layer2 in enumerate(tax_first_to_second[layer1]):
        tax_second_to_first[layer2] = layer1
        tmp2 =  tmp['children'][ix2]
        tax_second_to_third[layer2] = [t['name'] for t in tmp2['children']]
        # third layer
        for layer3 in tax_second_to_third[layer2]:
            tax_third_to_second[layer3] = layer2
        
#%%
skills_matches = {'budgeting': 'budget planning',
                  'project management': 'project planning and development skills',
                  'wind turbine technology': 'wind turbines',
                  'calculation': 'calculator',
                  'writing': 'report writing',
                  'planning':'project planning and development skills',
                  'autocad': 'modelling software',
                  '3d autocad': '3d modelling software'}



#%% 
# About Emsi skills
import requests
import pandas as pd
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
    
emsi_skills = obtain_skills_list()
# get the list
emsi_skills = emsi_skills['skills']
# remove certifications (they're likely to be USA specific anyway)
emsi_skills = [t for t in emsi_skills if t['type']!='Certification']
