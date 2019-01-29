# import numpy as np
# import pickle
#
# with open(r'I:\Inscopix_Data\all_mdict_nofilter.pickle', 'rb') as h:
#     mdict = pickle.load(h)
#
# percopen = (0.7, 0.8)
#
# for j in percopen:
#
#     for i in mdict.values():
#
#         print('Performing {} bootstrap...'.format(i.mouse))
#         bspref, bsdf = i.bootstrap_data_to_get_desired_open_arm_time(percopen=j, prctilethresh=95, filteropt=False)
#         file = r'I:\Inscopix_Data\Bootstrapped\{}_RTEPM_anymaze_inscopix_bootstrap_{}_nofilter.pickle'.format(i.mouse, j*100)
#
#         bootstrapdict = {
#             '{}_bspref'.format(i.mouse): bspref,
#             '{}_bsdf'.format(i.mouse): bsdf
#         }
#         with open(file, 'wb') as h:
#             pickle.dump(bootstrapdict, h)

import inscopixbox, pickle

bsdict = inscopixbox.batch_load_bootstrapped_data()

with open(r'I:\Inscopix_Data\all_mdict_nofilter.pickle', 'rb') as f:
    mdict = pickle.load(f)

inscopixbox.plot_bootstrapped_cell_pref(mdict, bsdict)

# inscopixbox.plot_perc_pref_open_vs_perc_spent_in_open(mdict)
#
# inscopixbox.plot_pref_idx_ecdf(mdict)