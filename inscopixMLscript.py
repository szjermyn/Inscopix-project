""" Run SVM and get best parameters for AUC """
import inscopixbox, pickle

f1_df, auc_df, mdict = inscopixbox.probe_SVM_with_different_C_and_gamma_values_nonlinear_kernel(fold=6, oversampleopt=False, niter=100)
bestparams = inscopixbox.get_best_SVM_parameters_for_each_mouse(auc_df, num=5)

with open(r'I:\Inscopix_Data\all_mdict_nofilter.pickle', 'rb') as h:
    mdict = pickle.load(h)

f1score, testy, all_y = mdict['m148'].run_svm_crossval(windowsize=40, n_comp=5, C=0.001, gamma=0.0025, kernel='rbf', fold=6)


for i in mdict.keys():
    mdict[i].centerexit_df = mdict[i].get_center_exit_categories('whole', 40)

SVM_df, mdict, errors = inscopixbox.probe_SVM_with_different_C_and_gamma_values_nonlinear_kernel()


SVM_df = MLdict['SVM_df']
mdict = MLdict['mdict']
errors = MLdict['errors']

bestparamdict = inscopixbox.get_best_SVM_parameters_for_each_mouse(SVM_df)



SVM_df, mdict, errors = inscopixbox.probe_SVM_with_different_C_values_linear_kernel()
SVM_df, mdict, errors = inscopixbox.probe_SVM_with_different_C_and_gamma_values_nonlinear_kernel()

#%%
import inscopixbox
import numpy as np
import re


Cvals = [10**i for i in range(-5,6)]
SVM_df, mdict, errors = inscopixbox.probe_SVM_with_different_C_values_linear_kernel(Cvals=Cvals)

inscopixbox.plot_percopen_vs_SVM_pred_accuracy(mdict, SVM_df)

windowsize = list(range(10,101,10))
maxaccmat = np.empty((len(windowsize), 17))


for w in range(len(windowsize)):
    SVM_df = inscopixbox.probe_SVM_with_different_C_and_gamma_values(windowsize=windowsize[w])
    midx = np.squeeze(np.where([bool(re.search('^m\d{2,4}', i)) for i in SVM_df.columns]))

    temp = np.asarray(SVM_df.iloc[:, midx])

    maxaccmat[w, :] = np.nanmax(temp, axis=0)



accdict = inscopixbox.plot_SVM_C_gamma_accuracy_matrix(SVM_df, True)


#%%
import pickle, re
import numpy as np
import pandas as pd

with open(r'I:\Inscopix_Data\all_mdict_nofilter.pickle', 'rb') as h:
    mdict = pickle.load(h)


Cvals = [10**i for i in range(-3,4)]
gammavals = [2*10**i for i in range(-6,1)]

columns = ['C', 'gamma', 'mean'] + list(mdict.keys())
index = range(len(Cvals)*len(gammavals))

SVM_df = pd.DataFrame(index=index, columns=columns)

midx = np.squeeze(np.where([bool(re.search('^m\d{2,4}', i)) for i in SVM_df.columns]))

idx = 0

for i in mdict.keys():
    mdict[i].centerexit_df = mdict[i].get_center_exit_categories('whole', 50)

for c in Cvals:

    for g in gammavals:

        print('Calculating values for C = {} and gamma = {}'.format(c, g))

        SVM_df.loc[idx, 'C'] = c
        SVM_df.loc[idx, 'gamma'] = g

        for i in mdict.keys():

            temp = mdict[i].run_svm_crossval(mdict[i].centerexit_df, windowsize=50, n_comp=5, C=c, gamma=g)
            if np.size(np.unique(temp[1])) == 2: #Only register results with both predictions
                SVM_df.loc[idx, i] = mdict[i].run_svm_crossval(mdict[i].centerexit_df, windowsize=50, n_comp=5, C=c, gamma=g)[0]

        SVM_df.loc[idx, 'mean'] = np.mean(SVM_df.iloc[idx, midx])

        idx += 1

#%% 1/2/19 SVM with different units

import pickle, inscopixbox
import numpy as np

with open(r'I:\Inscopix_Data\all_mdict_nofilter.pickle', 'rb') as f:
    mdict = pickle.load(f)

with open(r'I:\Inscopix_Data\ML\SVM_6fold_AUC_bestparams.pickle', 'rb') as f:
    bestparams = pickle.load(f)

percunit = np.arange(0.1, 1.1, 0.1)

for m in mdict.keys():

    params = bestparams[m]
    mouse = mdict[m]

    diffunits = mouse.run_svm_with_diff_unit_count(params, percunit)

    with open(r'I:\Inscopix_Data\ML\DiffUnits\{}.pickle'.format(m), 'wb') as f:
        pickle.dump(diffunits, f)


#%% 1/3/19 plot SVM score vs percentage of neurons

import inscopixbox, glob
import matplotlib.pyplot as plt

picklefiles = glob.glob(r'I:\Inscopix_Data\ML\DiffUnits\*.pickle')

allscores = []

plt.figure()
c = 1

for n, pf in enumerate(picklefiles):
    plt.subplot(2,2,c)
    allscores.append(inscopixbox.plot_SVM_score_vs_perc_cells(pf))

    if c % 4 == 0 and n != len(picklefiles):
        c = 1
        plt.figure()
    else:
        c += 1

picklefolder = r'I:\Inscopix_Data\ML\DiffUnits'
inscopixbox.batch_plot_SVM_score_vs_perc_cells(picklefolder)


 #%% 1/21/19 plot SVM score vs prefidx mean/std

import inscopixbox, glob

PIfile = r'I:\Inscopix_Data\m2225_RTEPM_halo_anymaze_inscopix.pickle'
SVMfile = r'I:\Inscopix_Data\ML\DiffUnits\m2225.pickle'

inscopixbox.plot_prediction_acc_vs_pref_idx_spread(SVMfile, PIfile)


#%% 1/22/19 SVM with different units

import pickle, inscopixbox, os
import numpy as np

with open(r'I:\Inscopix_Data\all_mdict_nofilter.pickle', 'rb') as f:
    mdict = pickle.load(f)

with open(r'I:\Inscopix_Data\ML\SVM_6fold_AUC_bestparams.pickle', 'rb') as f:
    bestparams = pickle.load(f)

for m in mdict.keys():

    outfile = r'I:\Inscopix_Data\ML\DiffGroups\mingroup\{}.pickle'.format(m)

    if os.path.isfile(outfile):
        continue
    # elif outfile == r'I:\Inscopix_Data\ML\DiffGroups\mingroup\m4513.pickle':
    #     continue



    print('\n\nProcessing {}...\n'.format(m))

    params = bestparams[m]
    mouse = mdict[m]

    try:
        diffgroups = mouse.run_svm_with_diff_subgroups(params, 'all')
    except:
        continue
    # diffgroups = mouse.run_svm_with_diff_subgroups(params, 'all')

    with open(outfile, 'wb') as f:
        pickle.dump(diffgroups, f)


#%%
import glob, inscopixbox

files = glob.glob(r'I:\Inscopix_Data\ML\DiffGroups\mingroup\*.pickle')

inscopixbox.plot_SVM_score_by_pref_idx_groups(files)
