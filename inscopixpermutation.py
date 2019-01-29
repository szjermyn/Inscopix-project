"""Create permutation test data for derivative preference index"""
import inscopixbox, glob, os, re, pickle

folder = r'I:\InscopixPaper'
files = glob.glob(os.path.join(folder, 'm*.pickle'))
outfolder = r'I:\InscopixPaper\PermutationTests'

for pf in files:

    real = inscopixbox.inscopix(pf, prctilethresh=95, cellsigopt='derivative', zscoreopt=True)
    temp = real.prefidx_permutation_test(niter=100, prctilethresh=95, filteropt=True, ignoreopt=True,
                                         shuffleopt='circular_fixed', cellsigopt='derivative', zscoreopt=True)

    mname = re.search('m\d{2,4}(?=_RTEPM)', pf).group(0)
    outfile = os.path.join(outfolder, mname + '_derivative_filter_permutation_test.pickle')

    with open(outfile, 'wb') as of:
        pickle.dump(temp, of)


# """Permutation test for SVM AUC"""
# import pickle
#
# with open(r'I:\Inscopix_Data\all_mdict_nofilter.pickle','rb') as f1:
#     mdict = pickle.load(f1)
#
# with open(r'I:\Inscopix_Data\ML\SVM_AUC_bestparams.pickle','rb') as f2:
#     bestparamsdict = pickle.load(f2)
#
# svm_auc_perm = {}
# svm_f1_perm = {}
#
# for i in mdict.keys():
#
#     print('\nProcessing {}...\n'.format(i))
#     svm_f1_perm[i], svm_auc_perm[i] = mdict[i].svm_permutation_test(bestparamsdict[i], fold=6)

# """SVM using best parameters and plot"""
# import pickle, inscopixbox
#
# with open(r'I:\Inscopix_Data\all_mdict_nofilter.pickle','rb') as f1:
#     mdict = pickle.load(f1)
#
# with open(r'I:\Inscopix_Data\ML\SVM_6fold_AUC_bestparams.pickle','rb') as f2:
#     bestparamsdict = pickle.load(f2)
#
# best_svm_auc = {}
# best_svm_f1 = {}
# pred_y = {}
# actual_y = {}
# score_y = {}
#
# for i in mdict.keys():
#
#     print('\nProcessing {}...\n'.format(i))
#     best_svm_f1[i], best_svm_auc[i], pred_y[i], actual_y[i], score_y[i] = mdict[i].run_svm_with_best_params(bestparamsdict[i], fold=6, svmiter=100)
#
# inscopixbox.plot_SVM_score_vs_percopen(mdict, best_svm_auc, True)
# inscopixbox.plot_closed_open_SVM_accuracy(pred_y, actual_y)


# """Plot permutation test for SVM AUC"""
# import pickle, inscopixbox
#
# with open(r'I:\Inscopix_Data\Permutation_Tests\SVM_AUC\svm_auc_perm.pickle','rb') as f:
#     svm_perm = pickle.load(f)
# with open(r'I:\Inscopix_Data\all_mdict_nofilter.pickle','rb') as f:
#     mdict = pickle.load(f)
#
# inscopixbox.plot_permutation_SVM_score_vs_percopen(mdict, svm_perm)

"""Plot permtuation test for preference index"""
import pickle, inscopixbox

folder = r'I:\Inscopix_Data\Permutation_Tests\Preference_index'
permdict = inscopixbox.batch_pickle2dict(folder, iden='m*derivative_filter_permutation*.pickle')

with open(r'I:\Inscopix_Data\all_mdict_filter_derivative.pickle', 'rb') as f:
    mdict = pickle.load(f)

inscopixbox.plot_permutation_cell_pref(mdict, permdict)

inscopixbox.plot_perc_pref_open_vs_perc_spent_in_open(mdict)