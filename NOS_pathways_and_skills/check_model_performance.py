#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:35:49 2019

@author: stefgarasto

This script is to evaluate the "accuracy" of a model to fit
educational/experience requirements.

"Accuracy" because it really matches how many we got right on the entire 
"training" dataset. Will probably need to be improved soon.

It used the confusion matrix and the F1 scores.

---------------------
In the workspace there needs to be the name of the target variable, 
as well as one of the following and the corresponding flag:

OPTION A
Flag_perf = 'bysocs' - A dictionary matching each 4-digit SOC to one of three 
education category. In this case, I haven't really trained the model (except for
a very low number of SOCs), so I can check on the full "training data". This
is only available when target_var is 'Edu'

OPTION B
Flag_perf = 'hybridcvmodel' - A dictionary matching some 4-digit SOC to one of the
education/experience category, a dictionary matching some titles to one of the 
edu/exp category, and a trained model classifying the full feature vectors. 
The problem here is that this is a hybrid model - The two dictionaries are obtained 
from a subset of the training dataset and tested on the rest, like the RF 
classifier (cross-validation).

OPTION C
Flag_perf = 'fullmodel' - Only a trained model that classifies the full feature
vector. This NEEDS to be derived from only a subset of the training data.


---------------------


"""

def plotCM(cm2plot, classes = ['Graduate', 'Pregraduate','Postgraduate'], 
           title = 'Confusion matrix', normalize = False):

    fig, ax = plt.subplots()
    im = ax.imshow(cm2plot, interpolation='nearest', cmap=sns.cm.rocket_r)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm2plot.shape[1]),
           yticks=np.arange(cm2plot.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm2plot.max() / 2.
    for i in range(cm2plot.shape[0]):
        for j in range(cm2plot.shape[1]):
            ax.text(j, i, format(cm2plot[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm2plot[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax


#%% imports
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
import copy

#%%
# select which model to evaluate
Flag_perf= 'hybridcvmodel' #'bysocs' #'hybridcvmodel'
WHICH_GLOVE = 'glove.6B.100d'
#target_var = 'Edu'
nfolds = 4
match_th = .9

outputfile = os.path.join(saveoutput, 
                         'model_performance_for_flag_{}_{}_{}.txt'.format(
                                 Flag_perf, WHICH_GLOVE, target_var))

outputfile_data = os.path.join(saveoutput, 
                    'model_performance_data_for_flag_{}_{}_{}.pickle'.format(
                                 Flag_perf,WHICH_GLOVE,target_var))

# set up the classes names
classes_dict = {'Edu': ['Pregraduate','Graduate','Postgraduate'],
                'Eduv2': ['Pregraduate','Graduate','Postgraduate'],
                'Exp': ['Entry-level','Junior','Senior','Expert'],
                'Exp3': ['Entry-level','Mid-level','Senior-level']}
classes = classes_dict[target_var]
n_classes = len(classes)


''' First build y_true and y_pred (targets and predictions)'''
dummy_strategies= ['stratified','most_frequent','prior','uniform']#,'constant']
clf_dummy = {}
y_pred_dummies = {}
SOCintersection = {}

#%%
if Flag_perf == 'bysocs':
    
    if not target_var in ['Edu','Eduv2']:
        raise ValueError('wrong target')
        
    with open(os.path.join(saveoutput,'government_based_model_Edu.pickle'), 
                                                                     'rb') as f:
        soc_to_edu_model2use = pickle.load(f)
    # extract the target values
    y_true = bgdatasmall[target_var].values
    # map all SOC codes to the corresponding education levels
    y_pred = bgdatasmall['SOC'].map(
            lambda x: soc_to_edu_model2use['{:.0f}'.format(x)])
    # extract the predicted values
    y_pred = y_pred.values

    # create a range of dummy classifiers
    for strategy in dummy_strategies:
        clf_dummy[strategy] = DummyClassifier(strategy=strategy)#, random_state=0)
        clf_dummy[strategy].fit(bgdatasmall['SOC'], bgdatasmall[target_var])
        y_pred_dummies[strategy] = clf_dummy[strategy].predict(bgdatasmall['SOC'])
        
    probabilities = []
    multiclass_probs = []
    df_model_soc = []
    df_model_title = []
    
elif Flag_perf == 'hybridcvmodel':

    #%load all the cross-validated models
    with open(os.path.join(saveoutput,
        '{}-fold-partition_{:.0f}_results_hybrid_rfc_fullcv_{}_{}.pickle'.format(
            nfolds,match_th*100,WHICH_GLOVE,target_var)),'rb') as f:
            cv_models = pickle.load(f)
    
    # If the cross-validation indices weren't saved, recompute them
    if 'train_indices' not in cv_models[0]:
        if cv_models[0]['extra_args']['CV'] == 'StratifiedKFold':
            cv = StratifiedKFold(n_splits=nfolds, 
                            shuffle =cv_models[0]['extra_args']['CVshuffle'], 
                            random_state = cv_models[0]['extra_args']['CVrs'])
            ix_cv = 0
            for train, test in cv.split(bgdatasmall['SOC'],bgdatasmall[target_var]):
                cv_models[ix_cv]['train_indices'] = train
                cv_models[ix_cv]['test_indices'] = test
                ix_cv += 1
        else:
            raise ValueError

    y_true = bgdatasmall[target_var].values
    
    '''I need to fill the predictions one cv-fold at a time'''
    # initialise the prediction column
    Target_pred = copy.deepcopy(bgdatasmall[target_var].map(lambda x: 'empty')) 
    Target_prob = copy.deepcopy(bgdatasmall['MeanSalary']*0)
    
    ''' Collect all normalised CM from the balanced cross-validated test sets'''
    all_CM_norm_blc = np.zeros((n_classes,n_classes,nfolds))
    CM_blc = np.zeros((n_classes,n_classes), dtype = np.int)
    CM_rfc = np.zeros((n_classes,n_classes), dtype = np.int)
    F1_blc = 0.0
    
    #%% iterate by the CV folds to fill in the whole dataset - each data point
    # is estimated from the fold in which it is NOT used as training data
    multiclass_probs = []
    for ix_cv in range(nfolds):
        #%
        cv_model = cv_models[ix_cv]
        #bgdata_train = bgdatasmall.iloc[cv_model['train']]
        # select test part
        bgdata_test = bgdatasmall.iloc[cv_model['test_indices']]
        # from the test part, extract the ones not classified by the MAP
        #TODO: check if I can reconstruct jobs_classified_test
        bgleft_test = bgdata_test[~cv_model['jobs_classified_test']]
        
        # select train part
        bgdata_train = bgdatasmall.iloc[cv_model['train_indices']]
        # from the test part, extract the ones not classified by the MAP
        bgleft_train = bgdata_train[~cv_model['jobs_classified_train']]
        
        #% first find how many SOCs from each suite we can classify
        tmp = [int(t) for t in list(cv_model['model_soc'].keys())]
        for SUPER in ['engineering','construction','management','financialservices']:
            tmp2 = [int(t) for t in soc4dist[SUPER]]
            if ix_cv == 0:
                SOCintersection[SUPER] = [len(set(tmp).intersection(tmp2))]
            else:
                SOCintersection[SUPER].append(len(set(tmp).intersection(tmp2)))
        
        # initialise the prediction column in the test dataset
        bgdata_test['Target_pred'] = 'empty'
        bgdata_test['Target_prob'] = 0
        # assign each known soc code to its match
        for isoc in cv_model['model_soc']:
            locations = bgdata_test['SOC']==isoc
            # assign the predicted category
            bgdata_test['Target_pred'][locations] = cv_model[
                    'model_soc'][isoc]['match']
            # assign the probability
            bgdata_test['Target_prob'][locations] = cv_model[
                    'model_soc'][isoc]['known accuracy']
        # assign each known job title to its match
        for isoc in cv_model['model_title']:
            locations = bgdata_test['title_processed']==isoc
            # assign the predicted category
            bgdata_test['Target_pred'][locations] = cv_model[
                    'model_title'][isoc]['match']
            # assign the probability
            bgdata_test['Target_prob'][locations] = cv_model[
                    'model_title'][isoc]['known accuracy']
        # at the end of this all the rows with ones in jobs_classified_test should
        # have been assigned
        
        # store all the results from the MAP phases
        if ix_cv == 0:
            df_model_soc = pd.DataFrame.from_dict(cv_model['model_soc'], 
                                    orient = 'index')
            df_model_title = pd.DataFrame.from_dict(cv_model['model_title'], 
                                    orient = 'index')
            df_model_soc = df_model_soc.rename({'match': 'match cv1',
                                        'known accuracy': 'known accuracy cv1'})
            df_model_title = df_model_title.rename({'match': 'match cv1',
                                        'known accuracy': 'known accuracy cv1'})
        else:
            df_model_soc = df_model_soc.join(pd.DataFrame.from_dict(
                    cv_model['model_soc'], orient = 'index'), 
                    rsuffix = ' cv{}'.format(ix_cv+1), how = 'outer')
            df_model_title = df_model_title.join(pd.DataFrame.from_dict(
                    cv_model['model_title'], orient = 'index'), 
                    rsuffix = ' cv{}'.format(ix_cv+1), how = 'outer')
        # recompute the encoding for the SOC codes
        #enc_left = preprocessing.OneHotEncoder(categories = 'auto', 
        # handle_unknown='ignore').fit(bgleft_train['SOC'].values.reshape(-1,1))
        # get the test features
        #x_test = get_all_features(bgleft_test, enc_left, enc_london)
        
        #% Now assign the rest based on the RFC. 
        #   First, predict values and probabilities
        B = cv_model['results']['ROS+RFC']['pred']
        multiclass_probs.append(cv_model['results']['ROS+RFC']['pred_probs'])
        #['pred_probs'])
        #['classifier'].predict_proba(x_test))
        
        # first, add the results from the model to the rows that have not been
        # classified via MAP
        bgdata_test['Target_pred'][~cv_model['jobs_classified_test']] = B
        # now save the probabilities. Note that from the multiclass 
        # probabilities select the highest one
        bgdata_test['Target_prob'][~cv_model['jobs_classified_test']
                                    ] = multiclass_probs[-1].max(axis = 1)
        # now all the rows in the bgdata_test have been classified and assigned 
        # a probability
        
        # assign the test rows to the main prediction series using the indices
        Target_pred.iloc[cv_model['test_indices']] = bgdata_test['Target_pred'].values
        Target_prob.iloc[cv_model['test_indices']] = bgdata_test['Target_prob'].values
        
        # Collect the normalised CM from the balanced version of the test data
        # It will be averaged at the end
        all_CM_norm_blc[:,:,ix_cv] = cv_model['results']['ROS+RFC'][
                'score_partial_blc']['CM_norm']
        CM_blc = CM_blc + cv_model['results']['ROS+RFC'][
                'score_partial_blc']['CM']
        F1_blc = F1_blc + cv_model['results']['ROS+RFC']['score_partial_blc'][
                'F1']/nfolds
        
        # retain the unbalanced confusion matrix for the RFC only
        CM_rfc = CM_rfc + cv_model['results']['ROS+RFC']['score_partial']['CM']
        
    #%%
    # stack the multiclass probabilities
    multiclass_probs = np.vstack(multiclass_probs)
    # at the end of this all the rows with ones in jobs_classified_test should
    # have been assigned
    y_pred = Target_pred.values
    y_true = bgdatasmall[target_var].values
    probabilities = Target_prob.values
    
    # take the average of the normalised and balanced CM
    CM_norm_blc = np.mean(all_CM_norm_blc, axis = 2)
    
    # create a range of dummy classifiers. NOTE that this will NOT depend on 
    # the model
    for strategy in dummy_strategies:
        clf_dummy[strategy] = DummyClassifier(strategy=strategy)#, random_state=0)
        clf_dummy[strategy].fit(bgdatasmall['SOC'], bgdatasmall[target_var])
        y_pred_dummies[strategy] = clf_dummy[strategy].predict(bgdatasmall['SOC'])
    
    
elif Flag_perf == 'fullmodel':
    raise NotImplementedError
else:
    raise ValueError('wrong model flag')


#%%
''' Use python inbuilt functions to compute performance metrics'''

# F1 score
F1w = f1_score(y_true, y_pred, labels=classes, average='weighted')
F1m = f1_score(y_true, y_pred, labels=classes, average='micro')
F1M = f1_score(y_true, y_pred, labels=classes, average='macro')

# confusion matrix
CM = confusion_matrix(y_true, y_pred, labels=classes)
CM_norm = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]

prfs = precision_recall_fscore_support(y_true, y_pred, labels = classes)
# now for the dummy predictions
F1w_dummies = {}
F1m_dummies = {}
F1M_dummies = {}
CM_dummies_norm = {}
for strategy in dummy_strategies:
    F1w_dummies[strategy] = f1_score(y_true, y_pred_dummies[strategy], 
              labels=classes, average='weighted')
    F1m_dummies[strategy] = f1_score(y_true, y_pred_dummies[strategy], 
              labels=classes, average='micro')
    F1M_dummies[strategy] = f1_score(y_true, y_pred_dummies[strategy], 
              labels=classes, average='macro')
    # confusion matrix
    CM_dummy = confusion_matrix(y_true, y_pred_dummies[strategy], 
                                labels=classes)
    CM_dummies_norm[strategy] = CM_dummy.astype('float'
                   ) / CM_dummy.sum(axis=1)[:, np.newaxis]


#%% print F1 and visualise the confusion matrix
Ntest= len(y_pred)
with open(outputfile, 'w') as f:
    print(''.join(['The micro, macro and weighted F1 score for the model ',
            'with flag \'{}\' are {:.3f}, {:.3f} and {:.3f}'.format(
            Flag_perf,F1m,F1M,F1w)]), file = f)
    print(''.join(['The macro F1 is the most demanding because is an unweighted ',
          'average of the F1 scores of each class. If a minority class is not ',
          'well predicted, it will decrease the macro score the most.']),file =f)
    
    if Flag_perf == 'hybridcvmodel':
        # add the average F1 from balanced data
        print('\n', file=f)
        print(''.join(['For the model with flag \'{}\', '.format(Flag_perf),
            'the average (cross-validated) F1 score for the trained RFC models ',
            'only, compute on balanced test data is {:.3f}'.format(F1_blc)]), 
            file = f)
            
    print('\n', file = f)
    for strategy in dummy_strategies:
        print(''.join(['The micro, macro and weighted F1 score for the dummy ',
                       'model with strategy {} are {:.3f}, {:.3f} and {:.3f}'.format(
            strategy,F1m_dummies[strategy],F1M_dummies[strategy],F1w_dummies[strategy])]),
            file = f)
    
    print('\n', file = f)
    print('The confusion matrix for the model with flag {} is:'.format(
            Flag_perf), file = f)
    print(np.around(CM_norm, 3), file = f)
    print('\n', file = f)
    correct_jobs = np.diag(CM).sum()
    print(''.join(['The number of jobs classified correctly is ',
                 '{} out of a total of {} ({:.2f}%))'.format(
                 correct_jobs,Ntest,100*correct_jobs/Ntest)]),
              file = f)
    if Flag_perf == 'hybridcvmodel':
        # print certainties for the whole data in intervals
        print('\n', file = f)
        print('Classification certainties for the whole dataset', file = f)
        for prob_th in [.9, .8, .7, .6, .5, .4]:
            print(''.join(['Number of job adverts classified with more than ',
                           '{:.1f}: {:.3f}'.format(
                prob_th, np.mean(probabilities>prob_th))]), file = f)
        # print certainties for "trained" portion of the model
        print('\n', file = f)
        print(''.join(['Of these, classification certainties for the data ',
                      'classified via RF are']), file = f)
        tmp_probs = multiclass_probs.max(axis = 1)
        for prob_th in [.9, .8, .7, .6, .5, .4]:
            print(''.join(['Number of job adverts classified with more than ',
                           '{:.1f}: {:.3f}'.format(
                prob_th, np.sum(tmp_probs>prob_th)/Ntest)]), file = f)
        print('\n', file= f)
        print('The nb. of data points classified via RF is {} ({:.2f}%)'.format(
                len(multiclass_probs), len(multiclass_probs)/Ntest), file = f)
        correct_jobs_rfc = np.diag(CM_rfc).sum()
        print('Of these, {} jobs ({:.2f}%) have been classified correctly'.format(
                correct_jobs_rfc, 100*correct_jobs_rfc/CM_rfc.sum()), file = f)
        print('\n', file = f)
        for SUPER in soc4dist.keys():
            if SUPER == 'other':
                continue
            print(''.join(['Across the {} folds, the number of SOC from {}'.format(
                    nfolds,SUPER), ' matched using MAP was: ']), file = f)
            print('({}, {}, {}, {}),'.format(*SOCintersection[SUPER]), 
                  'out of {} occupations'.format(len(soc4dist[SUPER])), file = f)
    
#%% now print on the standard output too
print(''.join(['The micro, macro and weighted F1 score for the model ',
            'with flag \'{}\' are {:.3f}, {:.3f} and {:.3f}'.format(
            Flag_perf,F1m,F1M,F1w)]))
for strategy in dummy_strategies:
    print('The F1 score for the dummy model with strategy {} is {:.3f}'.format(
        strategy,F1_dummies[strategy]))
print('The confusion matrix for the model with flag {} is:'.format(
        Flag_perf))
print(CM_norm)
print('\n')
print(''.join(['The number of jobs classified correctly is ',
                 '{} out of a total of {} ({:.2f}%)'.format(
                 correct_jobs,Ntest,100*correct_jobs/Ntest)]))

if Flag_perf == 'hybridcvmodel':
    print('\n')
    print('Classification certainties')
    for prob_th in [.9, .8, .7, .6, .5, .4]:
        print('Nb of job adverts classified with more than {:.1f}: {:.3f}'.format(
            prob_th, np.mean(probabilities>prob_th)))
    print('\n')
    print('Classification certainties for the data classified via RF')
    print('The number of data points classified via RF is {}'.format(
            len(multiclass_probs)))
    tmp_probs = multiclass_probs.max(axis = 1)
    for prob_th in [.9, .8, .7, .6, .5, .4]:
        print('Nb of job adverts classified with more than {:.1f}: {:.3f}'.format(
            prob_th, np.sum(tmp_probs>prob_th)/Ntest))


#%% Plots
SAVEFIG = True
# plot absolute confusion matrix
fig, ax = plotCM(CM, classes = classes, 
           title = 'Confusion matrix', normalize = False)
if SAVEFIG:
    fig.savefig(os.path.join(savefigures,
                    'confusion_matrix_for_model_flag_{}_{}_{}.png'.format(
                            Flag_perf,WHICH_GLOVE,target_var)), dpi= 200)
    
# plot normalised confusion matrix
fig, ax = plotCM(CM_norm, classes = classes, 
           title = 'Normalised confusion matrix', normalize = True)
if SAVEFIG:
    fig.savefig(os.path.join(savefigures,
        'normalised_confusion_matrix_for_model_flag_{}_{}_{}.png'.format(
                        Flag_perf, WHICH_GLOVE,target_var)), dpi= 200)

#%%
if Flag_perf == 'hybridcvmodel':
    # plot absolute confusion matrix from the balanced test data 
    # (which only includes the portion of data classified via RFC)
    fig, ax = plotCM(CM_blc, classes = classes, 
               title = 'Confusion matrix (balanced)', normalize = False)
    if SAVEFIG:
        fig.savefig(os.path.join(savefigures,
                        'confusion_matrix_blc_for_model_flag_{}_{}_{}.png'.format(
                                Flag_perf,WHICH_GLOVE,target_var)), dpi= 200)
        
    # plot normalised confusion matrix from the balanced test data 
    # (which only includes the portion of data classified via RFC)
    fig, ax = plotCM(CM_norm_blc, classes = classes, 
               title = 'Normalised confusion matrix (balanced)', normalize = True)
    if SAVEFIG:
        fig.savefig(os.path.join(savefigures,
            'normalised_confusion_matrix_blc_for_model_flag_{}_{}_{}.png'.format(
                    Flag_perf, WHICH_GLOVE,target_var)), dpi= 200)

    
    
#%% print the classification certainty (size of dominant class or trees-agreement)
if Flag_perf == 'hybridcvmodel':
    fig = plt.figure(figsize = (6,3.5))
    with sns.plotting_context('talk'):
        sns.distplot(probabilities, bins = np.arange(0,1.1,0.1), 
                 color = nesta_colours[3], kde = False,
                 hist_kws={"histtype": "stepfilled", "linewidth": 3,
                     "alpha": .75})#.plot()#color = nesta_colours[3])
        plt.ylabel('Number of job adverts')
        plt.xlabel('Classification certainty')
    plt.tight_layout()
    if SAVEFIG:
        fig.savefig(os.path.join(savefigures,
                'prediction_certainties_for_model_flag_{}_{}_{}.png'.format(
                            Flag_perf,WHICH_GLOVE,target_var)), dpi= 200)
#%%
SAVEDATA = True
if SAVEDATA:
    with open(outputfile_data, 'wb') as f:
        pickle.dump({'targets': y_true, 'predictions': y_pred, 'F1s': {'micro':F1m,
                        'macro': F1M, 'weighted': F1w, 'F1_blc': F1_blc},
                     'CM': CM,'CM_norm': CM_norm, 'F1_dummies': F1_dummies,
                     'CM_blc':CM_blc, 'CM_norm_blc':CM_norm_blc, 'CM_rfc':CM_rfc,
                     'dummy_strategies': dummy_strategies, 
                     'probabilities':probabilities, 'classes': classes,
                     'SOCintersection':SOCintersection,'df_model_soc':df_model_soc,
                     'df_model_title' : df_model_title}, f)
    
    if Flag_perf == 'hybridcvmodel':
        # print the matches from SOC codes and job titles as CSV
        df_model_soc.to_csv(os.path.join(saveoutput, 
                         'cv-model_soc2{}_for_flag_{}_{}.csv'.format(
                                 target_var.lower(),Flag_perf, WHICH_GLOVE)))
        df_model_title.to_csv(os.path.join(saveoutput, 
                         'cv-model_titles2{}_for_flag_{}_{}.csv'.format(
                                 target_var.lower(),Flag_perf, WHICH_GLOVE)))



