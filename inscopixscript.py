import inscopixbox

anymazetxtfile = r'I:\InscopixPaper\20160513_VIPhaloSG6m_RTEPM_m2049_anymazeData.txt'
insmatfile = r'I:\InscopixPaper\PCAICA_workspace_m2049_27-May-2016_WithTTLs.mat'
filename = r'I:\InscopixPaper\m2049_RTEPM_halo_anymaze_inscopix.pickle'

inscopixbox.pickle_anymaze_inscopix_data(anymazetxtfile, insmatfile, filename)


# """Create new mdicts"""
# import inscopixbox, pickle
#
# outfile = r'I:\Inscopix_Data\all_mdict_nofilter_nocorrderivative_nozscore.pickle'
# inscopixbox.batch_save_mdict(outfile, folder=r'I:\Inscopix_Data', threshtype='prctile', thresh=95, ignoreopt=True,
#                              filteropt=False, cellsigopt='derivative', zscoreopt=False)
#

# """Plot PI vs perc open"""
# import inscopixbox, pickle
# with open(r'I:\Inscopix_Data\all_mdict_nofilter_derivative.pickle', 'rb') as h:
#     mdict = pickle.load(h)
#
# inscopixbox.plot_perc_pref_open_vs_perc_spent_in_open(mdict)


#
# test = mdict['m2383'].calc_open_vs_closed_noise_corr(prefopt='open', plotopt=True)

# """Plot what's wrong with PI"""
# import inscopixbox, pickle
#
# with open(r'I:\Inscopix_Data\Permutation_Tests\Preference_index\m2383_permutation_test.pickle', 'rb') as f:
#     m2383perm = pickle.load(f)
#
# m2383 = inscopixbox.inscopix(r'I:\Inscopix_Data\m2383_RTEPM_halo_anymaze_inscopix.pickle', prctilethresh=95)
#
# m2383.plot_cell_perm_prefidx_vs_real_prefidx(m2383perm)

# """Plot example AUC"""
#
# import inscopixbox, pickle
# import numpy as np
# import matplotlib.pyplot as plt
#
# with open(r'I:\Inscopix_Data\Permutation_Tests\Preference_index\m325_permutation_test.pickle', 'rb') as f:
#     m325perm = pickle.load(f)
#
# m325 = inscopixbox.inscopix(r'I:\Inscopix_Data\m325_RTEPM_halo_anymaze_inscopix.pickle', prctilethresh=95)
#
#
# with open(r'I:\Inscopix_Data\ML\SVM_AUC_bestparams.pickle','rb') as f2:
#     bestparamsdict = pickle.load(f2)
# m325bestparams = bestparamsdict['m325']
#

# test = m325.run_svm_with_best_params(m325bestparams, fold=6, svmiter=10)
#
# bestsvmidx = 1
#
# sample_svm = test[4][:,bestsvmidx]
# closedidx = np.where(test[3] == 0)
# openidx = np.where(test[3])
#
# plt.figure()
# closedruns = sample_svm[closedidx]
# plt.scatter(closedruns, np.ones(closedruns.size),s=10, c='b')
# openruns = sample_svm[openidx]
# plt.scatter(openruns, np.ones(openruns.size),s=10, c='r')

#%%
import inscopixbox, glob, re, os
import matplotlib.pyplot as plt
import numpy as np

inscopixfiles = glob.glob(r'I:\Inscopix_Data\m*EYFP*inscopix.pickle')
mousedict={}
epochdict={}
percopendict={}

pvals = []

boundaryidx = np.linspace(0, 10800, 10, dtype=int)
startidx = boundaryidx[:-3]
endidx = boundaryidx[3:]


for ins in inscopixfiles:
    mousenum = re.search('(?<=Inscopix_Data\\\)m\d{3,4}', ins).group(0)
    mousedict[mousenum] = inscopixbox.inscopix(ins, prctilethresh=95, filteropt=False, cellsigopt='derivative')
    epochdict[mousenum] = mousedict[mousenum].calc_epoch_pref_idx(prctilethresh=95, filteropt=False, cellsigopt='derivative',
                                                                  startidx=startidx, endidx=endidx)[2]
    percopendict[mousenum] = mousedict[mousenum].calc_epoch_percentage_open_time(startidx=startidx, endidx=endidx)
    # pvals.append(inscopixbox.plot_prefidx_subsets(epochdict[mousenum][2]))
    # os.system('pause')


# for k, ed in zip(epochdict.keys(), epochdict.values()):
#     for i in range(len(ed[2])):
#         ed[2][i].to_csv(r'I:\Inscopix_Data\Epoch_Comparison\csvfiles\{}_{}_EYFP_raw.csv'.format(k, i))

