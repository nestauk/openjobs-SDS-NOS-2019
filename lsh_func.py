'''
LSH

LSH with MinHash is a technique that we can use to quickly
identify documents with a high overlap in content.
'''


from datasketch import MinHashLSHEnsemble, MinHash, MinHashLSH
import time
import pandas as pd

def print_elapsed(t0_local, task = 'current task'):
    print('Done with {}. Elapsed time: {:4f}'.format(task,time.time()-t0_local))

def shingles(text, char_ngram=5):
    '''
    This function splits strings into continuous sets of characters of length n. In the current example n = 5.
    '''
    if len(text) == 5:
        res = set([text, text])
    else:
        res = set(text[head:head + char_ngram] \
               for head in range(0, len(text) - char_ngram))
    return res

def perform_lsh(lsh_text, standard_labels, title_labels, char_ngram = 5, savefile = ''):
    t0 = time.time()
    shingled_desc = [shingles(desc) for desc in lsh_text]
    print_elapsed(t0, 'splitting the text into groups of characters')

    #Create hash signatures for shingles
    t0 = time.time()
    hash_objects = []
    for i in range(len(shingled_desc)):
        m = MinHash(num_perm=200)
        hash_objects.append(m)
    print_elapsed(t0, 'creating hash signatures')

    t0 = time.time()
    for ix, desc in enumerate(shingled_desc):
        for d in desc:
            hash_objects[ix].update(d.encode('utf8'))
    print_elapsed(t0, 'encoding hash objects')

    #Define LSH and Jaccard similarity threshold
    lsh = MinHashLSH(threshold=0.8, num_perm=200)

    content = []
    for ix, desc in enumerate(shingled_desc):
        content.append((standard_labels[ix], hash_objects[ix]))

    for ix,elem in enumerate(content):
        #lsh.insert('{}'.format(ix), elem[1]) #elem[0], elem[1])
        lsh.insert(elem[0], elem[1])

    #For each standard search all signatures and identify potential clashes (e.g. other standards with Jaccard similarity
    #of shingle sets greater or equal to the threshold). Note: some of the candidates might be false positives.
    candidates = {}
    for ix, desc in enumerate(shingled_desc):
        result = lsh.query(hash_objects[ix])
        if len(result) >1:
            candidates[standard_labels[ix] + ': ' + title_labels[ix]] = [(res, df_nos['Title'].loc[res]) for res in result]
            #candidates.append(result)
            print(standard_labels[ix] + ': ' + title_labels[ix], ': ', [(res, df_nos['Title'].loc[res]) for res in result])
            #print(standard_labels[ix], ': ',result)
            print('***************')
        else:
            candidates[standard_labels[ix]] = 'none'

    if len(savefile):
        pd.DataFrame.from_dict(candidates, orient = 'index').to_csv(savefile)
    return candidates, shingled_desc, content, lhs
