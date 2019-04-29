import numpy as np
import pandas as pd
import random as rn
import os
import json
import statsmodels.stats.api as sms
from scipy import stats

onevsone_file = 'results/onevsone.json'
onevsall_file = 'results/onevsall.json'
multiclass_file = 'results/multiclass.json'

with open(onevsone_file,'r') as f:
    onevsone = json.load(f)
with open(onevsall_file,'r') as f:
    onevsall = json.load(f)
with open(multiclass_file,'r') as f:
    multiclass = json.load(f)


##############################
# For each experiment calculate:
# 1. Accuracy
# 2. Lower and Upper 95% CI
# 3. p-value/t-value for t-test with naive baseline
# 4. p-value/t-value for t-test with lexico-syntactic baseline
###############################



multiclass_dict = {
                  'type':[],
                  'acc':[],
                  'L95':[],
                  'U95':[],
                   'p_baseline':[],
                  'p_lexico':[],
                    't_lexico':[],
                    't_baseline':[]}

for type in multiclass:
    multiclass_dict['type'].append(type)
    result_stats = sms.DescrStatsW(multiclass[type])

    #Calculate significance from baseline/naive and 95% CI
    acc = result_stats.mean
    l95, u95 = result_stats.tconfint_mean()
    l95 = np.maximum(l95, 0)
    u95 = np.minimum(u95, 1)
    t_baseline,p_baseline = result_stats.ttest_mean(1.0/16)[0:2]
    t_lexico,p_lexico = sms.ttest_ind(multiclass[type], multiclass['lexico_syntactic'])[0:2]

    multiclass_dict['acc'].append(acc)
    multiclass_dict['L95'].append(l95)
    multiclass_dict['U95'].append(u95)
    multiclass_dict['p_baseline'].append(p_baseline)
    multiclass_dict['p_lexico'].append(p_lexico)
    multiclass_dict['t_baseline'].append(t_baseline)
    multiclass_dict['t_lexico'].append(t_lexico)


onevsall_dict = {'author':[],
                  'type':[],
                  'acc':[],
                  'L95':[],
                  'U95':[],
                   'p_baseline':[],
                  'p_lexico':[],
                 't_lexico': [],
                 't_baseline': []}

for auth in onevsall:
    for type in multiclass:
        onevsall_dict['author'].append(auth)
        onevsall_dict['type'].append(type)

        # Calculate significance from baseline/naive and 95% CI
        result_stats = sms.DescrStatsW(onevsall[auth][type])
        acc = result_stats.mean
        l95, u95 = result_stats.tconfint_mean()
        l95 = np.maximum(l95, 0)
        u95 = np.minimum(u95, 1)
        t_baseline, p_baseline = result_stats.ttest_mean(.5)[0:2]
        t_lexico,p_lexico = sms.ttest_ind(onevsall[auth][type], onevsall[auth]['lexico_syntactic'])[0:2]

        onevsall_dict['acc'].append(acc)
        onevsall_dict['L95'].append(l95)
        onevsall_dict['U95'].append(u95)
        onevsall_dict['p_baseline'].append(p_baseline)
        onevsall_dict['p_lexico'].append(p_lexico)
        onevsall_dict['t_baseline'].append(t_baseline)
        onevsall_dict['t_lexico'].append(t_lexico)

onevsone_dict = {'author1':[],
                 'author2':[],
                  'type':[],
                  'acc':[],
                  'L95':[],
                  'U95':[],
                 'p_baseline': [],
                 'p_lexico': [],
                 't_lexico': [],
                 't_baseline': []}
for comb in onevsone:
    for type in multiclass:

        author1, author2 = comb.split('_')
        onevsone_dict['author1'].append(author1)
        onevsone_dict['author2'].append(author2)
        onevsone_dict['type'].append(type)

        # Calculate significance from baseline/naive and 95% CI
        result_stats = sms.DescrStatsW(onevsone[comb][type])
        acc = result_stats.mean
        l95, u95 = result_stats.tconfint_mean()
        l95 = np.maximum(l95, 0)
        u95 = np.minimum(u95, 1)
        t_baseline,p_baseline = result_stats.ttest_mean(.5)[0:2]
        t_lexico, p_lexico = sms.ttest_ind(onevsone[comb][type], onevsone[comb]['lexico_syntactic'])[0:2]

        onevsone_dict['acc'].append(acc)
        onevsone_dict['L95'].append(l95)
        onevsone_dict['U95'].append(u95)
        onevsone_dict['p_baseline'].append(p_baseline)
        onevsone_dict['p_lexico'].append(p_lexico)
        onevsone_dict['t_baseline'].append(t_baseline)
        onevsone_dict['t_lexico'].append(t_lexico)

#Write to CSV
pd.DataFrame(multiclass_dict).to_csv('results/multiclass.csv', index = False)
pd.DataFrame(onevsall_dict).to_csv('results/onevsall.csv', index = False)
pd.DataFrame(onevsone_dict).to_csv('results/onevsone.csv', index = False)