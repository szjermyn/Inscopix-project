import numpy as np
import pandas as pd
import copy, re, string, warnings, jsbox, pickle, os, glob
import matplotlib.pyplot as plt
import scipy.stats as sps
from matplotlib import colors
import matplotlib as mpl
import itertools as itools
import seaborn as sns
mpl.rcParams['pdf.fonttype'] = 42 # make pdf saveable


def pickle_anymaze_inscopix_data(anymazetxtfile, insmatfile, filename):

    anymazedata = parse_anymaze_txtfile(anymazetxtfile)
    anymazedata = format_anymaze_data(anymazedata)
    inscopixdata = load_inscopix_data(insmatfile)

    inscopixdict = {
        'anymazedata': pd.DataFrame.from_dict(anymazedata),
        'cell_sig': inscopixdata[0],
        'centroids': inscopixdata[1],
        'cell_seg': inscopixdata[2]
    }

    with open(filename, 'wb') as h:
        pickle.dump(inscopixdict, h)

def plot_ydiff_xdiff_and_find_tracking_errors(picklefile, pixelthresh=50):
    """ Check for anymaze screwups (instances where anymaze tracks the wire instead of the mouse, resulting in huge
    jumps in x-y positions"""

    with open(picklefile, 'rb') as h:
        tempdict = pickle.load(h)

    anymaze = tempdict['anymazedata']

    xpos = anymaze['CentrePosnX']
    ypos = anymaze['CentrePosnY']
    xdiff = np.nan_to_num(np.diff(xpos))
    ydiff = np.nan_to_num(np.diff(ypos))

    plt.figure()
    plt.scatter(xdiff, ydiff, 10)
    plt.show()

    return np.squeeze(np.where((abs(xdiff)+ abs(ydiff) > pixelthresh))) + 1, anymaze

def fix_anymaze_mistakes_and_pickle(picklefile, index):

    with open(picklefile, 'rb') as h:
        tempdict = pickle.load(h)
    anymaze = tempdict['anymazedata']

    for i in index:

        assert len(i) == 2
        startidx = i[0]-1
        endidx = i[1]+1

        xvals = np.asarray(anymaze.CentrePosnX.iloc[startidx:endidx])
        x = np.asarray(range(xvals.size))
        xp = [0, xvals.size]
        fp = [xvals[0], xvals[-1]]
        interpx = np.round(np.interp(x, xp, fp))

        yvals = np.asarray(anymaze.CentrePosnY.iloc[startidx:endidx])
        fp = [yvals[0], yvals[-1]]
        interpy = np.round(np.interp(x, xp, fp))

        anymaze.CentrePosnX.iloc[startidx:endidx] = interpx
        anymaze.CentrePosnY.iloc[startidx:endidx] = interpy

        closedbounds, approachbounds, openbounds, centerbounds = quick_define_epm_boundaries(anymaze)

        # reset all values for changed rows to 0 (except for InStimZone)
        colheaders = anymaze.columns.values
        colidx = np.squeeze(np.where([bool(re.search('^In', i)) for i in colheaders]))
        anymaze.iloc[i[0]:i[1], colidx[:-1]] = 0

        for j in np.arange(i[0],i[1]):
            posidx = np.asarray(anymaze.loc[j, ['CentrePosnX', 'CentrePosnY']])

            # check if closed using y position
            if (closedbounds[0,0] <= posidx[1] <= closedbounds[0,1]) or (closedbounds[1,0] <= posidx[1] <= closedbounds[1,1]):
                anymaze.loc[j, 'InClosedArms'] = 1
            elif (approachbounds[0,0] <= posidx[1] <= approachbounds[0,1]) or (approachbounds[1,0] <= posidx[1] <= approachbounds[1,1]):
                anymaze.loc[j, 'InClosedArms'] = 1
                anymaze.loc[j, 'InApproach'] = 1
            elif (openbounds[0,0] <= posidx[0] <= openbounds[0,1]) or (openbounds[1,0] <= posidx[0] <= openbounds[1,1]):
                anymaze.loc[j, 'InOpenArms'] = 1
            elif (centerbounds[0,0] <= posidx[0] <= centerbounds[0,1]) and (centerbounds[1,0] <= posidx[1] <= centerbounds[1,1]):
                anymaze.loc[j, 'InCenter'] = 1
            else:
                raise Exception('Something''s wrong!')

    tempdict['anymazedata'] = anymaze

    with open(picklefile, 'wb') as h:
        pickle.dump(tempdict, h)

    return anymaze

def quick_define_epm_boundaries(anymaze):

    # Get approch y-boundaries
    approachidx = anymaze.loc[anymaze['InApproach'] == 1, 'CentrePosnY']
    uniqueappidx = np.unique(approachidx)
    breakidx = np.where(np.diff(uniqueappidx) > 10)[0]
    approachbounds = np.asarray(((uniqueappidx[0], uniqueappidx[breakidx]), (uniqueappidx[breakidx+1],
                                                                             uniqueappidx[-1])),dtype=int)

    # Get closed arms y-boundaries
    closedidx = np.asarray(anymaze.loc[anymaze['InClosedArms'] == 1, 'CentrePosnY'])
    closedbounds = np.asarray(((np.nanmin(closedidx), approachbounds[0,0] - 1), (approachbounds[1,1] + 1,
                                                                                 np.nanmax(closedidx))), dtype=int)

    # Get open arms x-boundaries
    openidx = np.asarray(anymaze.loc[anymaze['InOpenArms'] == 1, 'CentrePosnX'])
    uniqueopenidx = np.unique(openidx)
    breakidx = np.where(np.diff(uniqueopenidx) > 10)[0]
    openbounds = np.asarray(((uniqueopenidx[0], uniqueopenidx[breakidx]), (uniqueopenidx[breakidx + 1],
                                                                             uniqueopenidx[-1])), dtype=int)

    # Get center x,y-boundaries
    centerbounds = np.asarray(((openbounds[0,1]+1, openbounds[1,0]-1), (approachbounds[0,1]+1,
                                                                        approachbounds[1,0]-1)), dtype=int)

    return closedbounds, approachbounds, openbounds, centerbounds

def parse_anymaze_txtfile(txtfile):

    with open(txtfile) as f:
        content = f.readlines()

    headers = content[0].strip('\n').split('\t')
    headers = [string.capwords(i).replace(' ', '') for i in headers]
    content = content[1:] # get rid of header
    content = [i.strip('\n').split('\t') for i in content]

    # anymazedata = {}
    anymazedata = []

    # for i in range(len(headers)):
    #     anymazedata[headers[i]] = [j[i] for j in content]

    for i in content:
        anymazedata.append({headers[j]:i[j] for j in range(len(headers))})

    return anymazedata

def format_anymaze_data(anymazedata):

    # timearray = anymazedata['Time']
    timearray = [i['Time'] for i in anymazedata]

    if ':' in timearray[0]:
        timearray = np.array([parse_anymaze_timestamps(i) for i in timearray])
    else:
        timearray = np.asarray(timearray, 'float')

    inscopixts = np.arange(0, 9*60, .05)

    idx = [(np.abs(timearray - i)).argmin() for i in inscopixts]

    ndata = [copy.deepcopy(anymazedata[i]) for i in idx]

    for i in range(len(ndata)):
        ndata[i]['Time'] = inscopixts[i]
        for j in ndata[i].keys():
            if ndata[i][j] == '':
                ndata[i][j] = None
            elif type(ndata[i][j]) is str:
                ndata[i][j] = float(ndata[i][j])
            else:
                continue

    # ndata = {i:[anymazedata[i][j] for j in idx] for i in anymazedata.keys()}
    # ndata['Time'] = list(inscopixts)
    #
    # for i in ndata.keys():
    #     if type(ndata[i][-1]) is str:
    #         ndata[i] = [None if not j else int(j) for j in ndata[i]]
    #     else:
    #         continue

    return ndata


def load_inscopix_data(insmatfile, trimopt=True):

    temp = jsbox.loadmat(insmatfile)
    cell_sig = temp['cell_sig']
    centroids = temp['segcentroid']
    cell_segments = temp['ica_segments']

    if trimopt:
        cell_sig = cell_sig[:,0:10800]

    return cell_sig, centroids, cell_segments


def save_anymaze_data_to_json(anymazedict, outfile):
    import json

    if not '.json' in outfile:
        outfile = outfile + '.json'

    with open(outfile, 'w') as jfile:
        json.dump(anymazedict, jfile, indent=4)

    print('\nData saved to %s\n' % outfile)


def parse_anymaze_timestamps(timestamp):
    time, ms = timestamp.split('.')
    ms = round(float(ms)/100,2)
    h, m, s = map(float, time.split(':'))

    return h*60*60 + m*60 + s + ms # timestamp in seconds


class inscopix(object):

    def __init__(self, file, stdthresh=(), prctilethresh=(), ignoreopt=True, filteropt=False, cellsigopt='raw', zscoreopt=True):

        warnings.simplefilter('ignore', RuntimeWarning)

        fileext = os.path.splitext(file)[1]

        print('\nLoading {}...'.format(file))

        if fileext == '.mat':
            matvars = jsbox.loadmat(file)
            cellsig = matvars['cell_sig']
            self.cellsegments = matvars['cell_segments']
            temp = matvars['anymazedata']
            fn = temp[0]._fieldnames
            anymaze = pd.DataFrame.from_dict({i: np.asarray([getattr(j, i) for j in temp]) for i in fn})

        elif fileext == '.pickle':
            with open(file, 'rb') as h:
                temp = pickle.load(h)
            anymaze = temp['anymazedata']
            cellsig = temp['cell_sig']
            cellseg = temp['cell_seg']
            cellseg[cellseg > 0] = 1    # convert all positive values to 1
            self.cellsegments = cellseg
            self.centroids = temp['centroids']

        else:
            raise Exception('File extention not recognized')


        if not stdthresh and not prctilethresh:
            raise Exception('Enter a value for stdthresh or prctilethresh')
        elif np.isscalar(stdthresh) and np.isscalar(prctilethresh):
            raise Exception('Enter a value for either stdthresh or prctilethresh only')

        # anymaze = self.anymaze
        # cellsig = self.cellsig

        if ignoreopt:
            startthresh = 10  # ignore the first 10 seconds of experiment
            startidx = int(np.squeeze(np.where(anymaze['Time'] == startthresh)))
        else:
            startidx = 0

        self.mouse = re.search('m\d{2,4}', file).group(0)

        assert (np.size(cellsig, 1) == len(anymaze['Time'])), "Cellsig and anymaze must have the same number of samples!"

        print('Creating plusmaze map for {}...'.format(self.mouse))
        anymaze, plusmazemap, posdf, pos, maxxpos, maxypos = create_plusmazemap(anymaze, startidx)

        heatmap = np.zeros((maxypos + 10, maxxpos + 10))

        for i in range(startidx, pos.shape[1]):
            heatmap[pos[0, i], pos[1, i]] += 1

        # Normalize cell sig if necessary
        if zscoreopt:
            zcellsig = sps.zscore(cellsig, 1)
        else:
            zcellsig = copy.deepcopy(cellsig)

        if cellsigopt == 'derivative':
            proc_cellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
            proc_cellsig[proc_cellsig<0] = 0
        elif cellsigopt == 'basic_derivative':
            proc_cellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
        elif cellsigopt == 'derivative_product':
            dcellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
            dcellsig[dcellsig <= 0] = 0
            proc_cellsig = zcellsig * dcellsig
        elif cellsigopt == 'raw':
            proc_cellsig = zcellsig
        elif cellsigopt == 'raw_product':
            dcellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
            dcellsig[dcellsig <= 0] = 0
            dcellsig[dcellsig > 0] = 1
            proc_cellsig = zcellsig * dcellsig
        elif cellsigopt == 'df':
            blcellsig = calc_inscopix_trace_running_mean(zcellsig, window=200)
            proc_cellsig = zcellsig - blcellsig
        elif cellsigopt == 'df/f':
            blcellsig = calc_inscopix_trace_running_mean(zcellsig, window=200)
            assert(all(np.min(blcellsig, axis=1)>0))
            proc_cellsig = (zcellsig - blcellsig) / blcellsig
        else:
            raise Exception('Invalid cellsigopt!')

        # Get individual cell's calciummap
        calciummap, calciummap_thresh, norm_cmap = \
            calculate_calciummaps(proc_cellsig[:, startidx:], heatmap, pos[:, startidx:], prctilethresh, stdthresh, filteropt)

        self.anymaze_raw = anymaze
        self.calciummap = calciummap
        self.calciummap_norm = norm_cmap
        self.calciummap_thresh = calciummap_thresh
        self.plusmazemap = plusmazemap
        self.heatmap = heatmap
        self.posdf = posdf
        self.rawcellsig = cellsig
        print('Classifying cells...')
        self.prefdf = classify_cells_by_sig_pixels(posdf, calciummap_thresh)

        self.cellsigraster = convert_cellsig_to_raster(cellsig, prctilethresh=95, timethresh=500, binsize=50)
        # self.anymaze_ts = process_anymaze_timeseries(anymaze, simplifyopt=1)
        print('Computing anymaze timeseries...')
        anymaze_ts = process_anymaze_timeseries(anymaze, simplifyopt=0)
        self.percopen = anymaze_ts.loc[(anymaze_ts['status'] == 'center') | (anymaze_ts['status'] == 'open')].shape[0] / anymaze_ts.shape[0]
        print('Calculating mouse speed...')
        self.anymaze_ts = calc_mouse_speed_from_anymaze_data(anymaze_ts, sigma=3)
        self.processed_cellsig = proc_cellsig


    def prefidx_permutation_test(self, niter=100, prctilethresh=(), stdthresh=(), filteropt=False, ignoreopt=True,
                                 shuffleopt='circular_fixed', cellsigopt='raw', zscoreopt=True):

        if zscoreopt:
            zcellsig = sps.zscore(self.rawcellsig, axis=1)
        else:
            zcellsig = self.rawcellsig

        if ignoreopt:
            startidx = 200
        else:
            startidx = 0

        nbins = zcellsig.shape[1]

        if shuffleopt == 'circular_fixed':
            shiftidx = np.round(np.linspace(nbins/niter, nbins - nbins/niter, niter))


        if cellsigopt == 'derivative':
            proc_cellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
            proc_cellsig[proc_cellsig<0] = 0
        elif cellsigopt == 'derivative_product':
            dcellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
            dcellsig[dcellsig <= 0] = 0
            proc_cellsig = zcellsig * dcellsig
        elif cellsigopt == 'raw':
            proc_cellsig = zcellsig
        elif cellsigopt == 'raw_product':
            dcellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
            dcellsig[dcellsig <= 0] = 0
            dcellsig[dcellsig > 0] = 1
            proc_cellsig = zcellsig * dcellsig
        else:
            raise Exception('cellsigopt is invalid!')


        permprefidx = np.empty((niter, proc_cellsig.shape[0]))

        for i in range(niter):

            print('\nIteration {} of {}'.format(i+1, niter))

            if shuffleopt == 'circular_fixed':
                shuffcellsig = circularly_shuffle_cellsig(proc_cellsig, shiftidx[i])
            elif shuffleopt == 'circular_random':
                shuffcellsig = circularly_shuffle_cellsig(proc_cellsig)
            elif shuffleopt == 'random':
                shuffcellsig = np.random.permutation(proc_cellsig.T).T
            else:
                raise Exception('shuffleopt is invalid!')


            __, calciummap_thresh, __ = \
                calculate_calciummaps(shuffcellsig, self.heatmap, self.anymaze_raw, prctilethresh=prctilethresh,
                                      stdthresh=stdthresh, filteropt=filteropt)

            prefdf = classify_cells_by_sig_pixels(self.posdf, calciummap_thresh)
            permprefidx[i,:] = prefdf.loc[:, 'prefidx']

        return permprefidx

    def prefidx_permutation_test_by_epoch(self, niter=100, prctilethresh=(), stdthresh=(), filteropt=False,
                                          ignoreopt=True, shuffleopt='circular_fixed', cellsigopt='raw', zscoreopt=True):


        if ignoreopt:
            startidx = 200
        else:
            startidx = 0

        cellsig = self.rawcellsig
        nbins = cellsig.shape[1]

        if shuffleopt == 'circular_fixed':
            shiftidx = np.round(np.linspace(nbins / niter, nbins - nbins / niter, niter))


        permprefidx = np.empty((niter, cellsig.shape[0]))

        for i in range(niter):

            print('\nIteration {} of {}'.format(i + 1, niter))

            if shuffleopt == 'circular_fixed':
                shuffcellsig = circularly_shuffle_cellsig(cellsig, shiftidx[i])
            elif shuffleopt == 'circular_random':
                shuffcellsig = circularly_shuffle_cellsig(cellsig)
            elif shuffleopt == 'random':
                shuffcellsig = np.random.permutation(cellsig.T).T
            else:
                raise Exception('shuffleopt is invalid!')

            __,__,permprefidx_epoch = calc_epoch_pref_idx(shuffcellsig, self.anymaze, self.posdf, stdthresh=stdthresh,
                                                        prctilethresh=prctilethresh, filteropt=filteropt,
                                                        zscoreopt=zscoreopt, cellsigopt=cellsigopt)

            if zscoreopt:
                zcellsig = sps.zscore(self.shuffcellsig, axis=1)
            else:
                zcellsig = self.shuffcellsig

            if cellsigopt == 'derivative':
                proc_cellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
                proc_cellsig[proc_cellsig < 0] = 0
            elif cellsigopt == 'derivative_product':
                dcellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
                dcellsig[dcellsig <= 0] = 0
                proc_cellsig = zcellsig * dcellsig
            elif cellsigopt == 'raw':
                proc_cellsig = zcellsig
            elif cellsigopt == 'raw_product':
                dcellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
                dcellsig[dcellsig <= 0] = 0
                dcellsig[dcellsig > 0] = 1
                proc_cellsig = zcellsig * dcellsig
            else:
                raise Exception('cellsigopt is invalid!')

            pos = np.array(self.anymaze.loc[:, ['CentrePosnY', 'CentrePosnX']]).T
            temppos = pos[:,startidx:]

            __, calciummap_thresh, __ = \
                calculate_calciummaps(proc_cellsig[:,startidx:], self.heatmap, temppos, prctilethresh=prctilethresh,
                                      stdthresh=stdthresh, filteropt=filteropt)

            prefdf = classify_cells_by_sig_pixels(self.posdf, calciummap_thresh)
            permprefidx[i, :] = prefdf.loc[:, 'prefidx']

        return permprefidx, permprefidx_epoch

    def plot_intercell_distance_vs_correlation(self):

        zcellsig = sps.zscore(self.rawcellsig, axis=1)
        corrmat = np.corrcoef(zcellsig)

        centroids = self.centroids
        distmat = np.zeros((np.size(centroids,0),np.size(centroids,0)))
        comb = np.asarray(list(itools.combinations(range(np.size(centroids,0)), 2)))

        for i in comb:
            dist = centroids[i[0],:] - centroids[i[1],:]
            distmat[i[0], i[1]] = np.sqrt(dist[0]**2 + dist[1]**2)

        corrvec = np.triu(corrmat, 1)
        corrvec = corrvec[corrvec != 0]

        distvec = distmat[distmat != 0]

        plt.figure()
        plt.scatter(distvec, corrvec, s=8)
        regres = sps.linregress(distvec, corrvec)

        f = lambda x: regres.slope*x + regres.intercept
        xedges = np.asarray(plt.xlim(), int)
        plt.plot(xedges, f(xedges), 'k--')
        plt.xlabel('Distance (pixels)')
        plt.ylabel('Correlation value')
        plt.figtext(.6,.8,'$r^2$ = {:.2e}\np = {:.2f}'.format(regres.rvalue**2, regres.pvalue))
        return regres

    def plot_plusmazemap(self, plotopt='all'):

        if plotopt == 'all':
            cmap = colors.ListedColormap(['w','b','r','c','m'])
            norm = colors.BoundaryNorm([0,1,2,3,4,5], cmap.N)
        elif plotopt == 'open/closed':
            cmap = colors.ListedColormap(['w', 'b', 'r'])
            norm = colors.BoundaryNorm([0, 1, 3, 5], cmap.N)
        else:
            raise Exception('Plotopt should be ''all'' or ''open/closed''')
        plt.figure()
        plt.imshow(self.plusmazemap, cmap=cmap, norm=norm, interpolation='none')
        plt.ylabel('y position')
        plt.xlabel('x position')
        plt.colorbar()
        plt.show()

    def plot_heatmap(self):

        from palettable.colorbrewer.sequential import Reds_9 as R9

        heatmap = self.heatmap
        heatmap[heatmap==0] = np.nan
        plt.figure()
        plt.imshow(self.heatmap, R9.mpl_colormap)
        plt.ylabel('y position')
        plt.xlabel('x position')
        plt.colorbar()
        plt.show()

    def plot_cell_heatmap(self, maptype='thresh'):

        from palettable.colorbrewer.sequential import Reds_9 as R9
        plt.figure()
        c = 1
        prefidx = self.prefdf.loc[:,'prefidx']

        if maptype == 'thresh':
            cmap = self.calciummap_thresh
            cmap[cmap != 0] = 1
        elif maptype == 'norm':
            cmap = self.calciummap_norm
        elif maptype == 'raw':
            cmap = self.calciummap
        else:
            raise Exception('maptype must be ''thresh'', ''norm'' or ''raw''')

        cmap[cmap == 0] = np.nan

        for i in range(np.size(cmap, axis=0)):
            plt.subplot(2, 3, c)
            if maptype == 'thresh':
                colmap = colors.ListedColormap(['w', 'k'])
                norm = colors.BoundaryNorm([0, 1, 2], colmap.N)
                plt.imshow(cmap[i], cmap=colmap, norm=norm, interpolation=None)
            else:
                plt.imshow(cmap[i], R9.mpl_colormap)
            plt.ylabel('y position')
            plt.xlabel('x position')
            plt.title('Unit {}, P.I. = {:.2f}'.format(i, prefidx[i]))

            if c == 6:
                plt.figure()
                # figManager = plt.get_current_fig_manager()
                # figManager.window.showMaximized()
                c = 1
            else:
                c = c + 1

        plt.show()

    def plot_zscored_cellsig(self, xrange=()):

        cellsig = self.rawcellsig
        zcellsig = sps.zscore(cellsig, 1)
        plt.figure()
        c = 1

        for i in range(np.size(zcellsig, axis=0)):
            plt.subplot(2,3,c)
            plt.plot(zcellsig[i,:])
            plt.ylabel('z-scored activity')
            plt.xlabel('Sample number')
            plt.title('Unit {}'.format(i))

            if xrange:
                plt.xlim(xrange)

            if c == 6:
                plt.figure()
                c = 1
            else:
                c = c + 1

        plt.show()

    def plot_summary_figures(self):

        prefdf = self.prefdf
        prefdict = prefdf.preference
        prefidx = prefdf.prefidx
        cellsegments = self.cellsegments

        # Plot histogram
        plt.figure()
        plt.hist(prefidx, np.linspace(-1, 1, 21), edgecolor='k', linewidth=1)
        plt.axvline(x=np.mean(prefidx), linestyle='--', color='r')
        plt.xlim((-1, 1))
        plt.xlabel('Preference index')
        plt.ylabel('Count')
        plt.title('Histogram of preference index values')

        prefidxplot = np.zeros(np.shape(cellsegments)[1:])
        prefcatplot = np.zeros(np.shape(cellsegments)[1:])

        # Categorical index
        prefdict.replace('nopref', 1, inplace=True)
        prefdict.replace('closed', 2, inplace=True)
        prefdict.replace('open', 3, inplace=True)

        for i in range(np.size(cellsegments, 0)):
            # preference index calculation
            mapidx = np.transpose(np.nonzero(prefidxplot))
            cellsegidx = np.transpose(np.nonzero(cellsegments[i]))

            # This whole part is for averaging overlapping pixels (for preference index plot) and taking max values for overlapping pixels (for categorical plot)
            intidx = tuple(zip(*[x for x in set(tuple(x) for x in mapidx) and set(
                tuple(x) for x in cellsegidx)]))  # Get intersection of indices
            nonintidx = tuple(zip(*[x for x in set(tuple(x) for x in cellsegidx) - set(
                tuple(x) for x in mapidx)]))  # Get set difference of indices

            prefidxplot[intidx] = np.mean(np.stack((cellsegments[i][intidx] * prefidx[i], prefidxplot[intidx])), axis=0)
            prefidxplot[nonintidx] = cellsegments[i][nonintidx] * prefidx[i]

            # categorical mapping
            prefcatplot[intidx] = np.max(np.stack((cellsegments[i][intidx] * prefdict[i], prefcatplot[intidx])), axis=0)
            prefcatplot[nonintidx] = cellsegments[i][nonintidx] * prefdict[i]

        # Trim both plots
        nonzeroidx = np.transpose(np.nonzero(prefidxplot))
        miny = np.min(nonzeroidx[:,0]) - 10
        maxy = np.max(nonzeroidx[:,0]) + 10
        minx = np.min(nonzeroidx[:,1]) - 10
        maxx = np.max(nonzeroidx[:,1]) + 10

        # Plot sig plot
        from palettable.colorbrewer.diverging import RdBu_11_r as RB11

        plt.figure()
        plt.imshow(prefidxplot, cmap=RB11.mpl_colormap, vmin=-1, vmax=1)
        plt.colorbar()
        plt.xlim((minx, maxx))
        plt.ylim((miny, maxy))
        plt.xlabel('Imaging window x position')
        plt.ylabel('Imaging window y position')
        plt.show()


        # Plot categorical plot
        plt.figure()
        cmap = colors.ListedColormap(['w', (.5, .5, .5), 'b', 'r'])
        norm = colors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)
        plt.imshow(prefcatplot, cmap=cmap, norm=norm, interpolation=None)
        plt.colorbar()
        plt.xlim((minx, maxx))
        plt.ylim((miny, maxy))
        plt.xlabel('Imaging window x position')
        plt.ylabel('Imaging window y position')
        plt.show()

    def plot_speed_cellsig_xcorr(self, maxlag=3):

        speed = self.anymaze_ts['speed']
        cellsig = self.rawcellsig
        zcellsig = sps.zscore(cellsig, axis=1)

        fs = 50 #sampling rate
        maxlagsamples = int(maxlag * 1000 / fs)

        plt.figure()
        c = 1

        for i in range(np.size(zcellsig, 0)):
            plt.subplot(2, 3, c)
            plt.xcorr(speed, zcellsig[i,:], maxlags=maxlagsamples)
            plt.ylabel('Correlation')
            locs, __ = plt.xticks()
            labels = locs / (1000/fs)
            plt.xticks(locs, labels)
            plt.xlim([-maxlagsamples,maxlagsamples])
            plt.title('Unit {}'.format(i))

            if c == 6 and i != np.size(zcellsig, 0):
                plt.show()
                plt.figure()
                # figManager = plt.get_current_fig_manager()
                # figManager.window.showMaximized()
                c = 1
            else:
                c = c + 1

        plt.show()

    def plot_inscopix_traces(self, selectopt=()):

        cellsig = self.rawcellsig

        if selectopt and selectopt < np.size(cellsig):
            randidx = np.sort(np.random.choice(np.size(cellsig, 0), selectopt, replace=False))
            zcellsig = sps.zscore(cellsig[randidx, :], 1)
        else:
            zcellsig = sps.zscore(cellsig, 1)

        plt.figure()
        timeaxis = np.round(np.linspace(0, 9 * 60, np.size(zcellsig, 1)), 2)

        for i in range(np.size(zcellsig, 0)):
            plt.plot(timeaxis, zcellsig[i] + i * 5)

        plt.xlim((0, 9 * 60))
        plt.xlabel('Time (s)')
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left='off',  # ticks along the bottom edge are off
            right='off',  # ticks along the top edge are off
            labelleft='off')  # labels along the bottom edge are off

    def plot_pref_piechart(self):

        pref = self.prefdf['preference']
        noprefnum = np.size(pref[pref=='nopref'])
        opennum = np.size(pref[pref=='open'])
        closednum = np.size(pref[pref=='closed'])

        labels = 'no preference', 'open', 'closed'
        colors = ['grey','lightcoral','lightskyblue']


        plt.figure()
        plt.pie((noprefnum, opennum, closednum), labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title('{} cell preference'.format(self.mouse))


    def calc_epoch_pref_idx(self, stdthresh=(), prctilethresh=(), filteropt=False, zscoreopt=True, cellsigopt='raw',
                            startidx=(), endidx=()):

        if not stdthresh and not prctilethresh:
            raise Exception('Enter a value for stdthresh or prctilethresh')
        elif np.isscalar(stdthresh) and np.isscalar(prctilethresh):
            raise Exception('Enter a value for either stdthresh or prctilethresh only')

        warnings.simplefilter('ignore', RuntimeWarning)
        cellsig = self.rawcellsig
        anymaze = self.anymaze_raw
        posdf = self.posdf

        # Process some basic information from create_plusmazemap
        pos = np.array(anymaze.loc[:,['CentrePosnY','CentrePosnX']]).T
        maxypos = np.max(pos[0,:])
        maxxpos = np.max(pos[1,:])

        if not any(startidx) and not any(endidx):
            boundaryidx = np.linspace(0, np.size(cellsig, 1), 4, dtype=int)
            startidx = boundaryidx[:-1]
            endidx = boundaryidx[1:]
            startthresh = 10  # ignore the first 10 seconds of experiment
            startidx[0] = int(np.squeeze(np.where(anymaze['Time'] == startthresh)))

        print('Creating plusmazemap for epochs...')

        # anymaze, plusmazemap, posdf, pos, maxxpos, maxypos = create_plusmazemap(anymaze, startidx)

        calciummap = []
        calciummap_thresh = []
        prefdf = []

        for i in range(startidx.size):

            print('\nProcessing epoch {}'.format(i+1))

            # if i != 0:
            #     startidx = boundaryidx[i]
            #
            # endidx = boundaryidx[i+1]


            # Normalize cell sig if necessary
            if zscoreopt:
                zcellsig = sps.zscore(cellsig[:,startidx[i]:endidx[i]], 1)
            else:
                zcellsig = cellsig[:,startidx[i]:endidx[i]]

            if cellsigopt == 'derivative':
                proc_cellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
                proc_cellsig[proc_cellsig < 0] = 0
            elif cellsigopt == 'basic_derivative':
                proc_cellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
            elif cellsigopt == 'derivative_product':
                dcellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
                dcellsig[dcellsig <= 0] = 0
                proc_cellsig = zcellsig * dcellsig
            elif cellsigopt == 'raw':
                proc_cellsig = zcellsig
            elif cellsigopt == 'raw_product':
                dcellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
                dcellsig[dcellsig <= 0] = 0
                dcellsig[dcellsig > 0] = 1
                proc_cellsig = zcellsig * dcellsig
            elif cellsigopt == 'df':
                blcellsig = calc_inscopix_trace_running_mean(zcellsig, window=200)
                proc_cellsig = zcellsig - blcellsig
            elif cellsigopt == 'df/f':
                blcellsig = calc_inscopix_trace_running_mean(zcellsig, window=200)
                assert (all(np.min(blcellsig, axis=1) > 0))
                proc_cellsig = (zcellsig - blcellsig) / blcellsig
            else:
                raise Exception('Invalid cellsigopt!')

            temppos = pos[:,startidx[i]:endidx[i]]

            heatmap = np.zeros((maxypos + 10, maxxpos + 10))

            for j in range(temppos.shape[1]):
                heatmap[temppos[0, j], temppos[1, j]] += 1

            calciummaps = calculate_calciummaps(proc_cellsig, heatmap, temppos, prctilethresh=prctilethresh,
                                                filteropt=filteropt, maxxpos=maxxpos, maxypos=maxypos)

            calciummap.append(calciummaps[0])
            calciummap_thresh.append(calciummaps[1])

            prefdf.append(classify_cells_by_sig_pixels(posdf, calciummap_thresh[i]))

        return calciummap, calciummap_thresh, prefdf

    def calc_epoch_percentage_open_time(self, startidx=(), endidx=()):

        anymaze_ts = self.anymaze_ts

        if not any(startidx) and not any(endidx):
            boundaryidx = np.linspace(0, np.size(cellsig, 1), 4, dtype=int)
            startidx = boundaryidx[:-1]
            endidx = boundaryidx[1:]
            startthresh = 10  # ignore the first 10 seconds of experiment
            startidx[0] = int(np.squeeze(np.where(anymaze['Time'] == startthresh)))

        assert(startidx.size == endidx.size)

        perc_open = np.empty((startidx.size, 1))

        for i in range(startidx.size):
            status_count = anymaze_ts.loc[startidx[i]:endidx[i], 'status'].value_counts()
            perc_open[i] = np.sum(status_count[['open','center']]) / np.sum(status_count)

        return perc_open


    def get_center_exit_categories(self, tstype='whole', tslength=20):

        assert(tstype=='snippets' or tstype=='whole')

        # get position-status time series and do run-length encoding
        ts = self.anymaze_ts
        centerexit_df = get_run_sequences(ts, tstype, tslength)

        return centerexit_df

    def run_svm_crossval(self, windowsize=40, n_comp=5, C=1, gamma='auto', kernel='rbf', fold=1, niter=10, oversampleopt=False):

        """If fold is 1, run leave-one-out SVM. Otherwise, divide data into number of groups as specified by fold."""

        assert fold >= 1

        # remove values smaller than 0
        try:
            centerexit_df = self.centerexit_df
        except AttributeError:
            centerexit_df = self.get_center_exit_categories('whole', windowsize)

        rawcellsig = self.rawcellsig

        f1score, auc, testy, all_y, score_y = run_inscopix_svm(
            centerexit_df, rawcellsig, windowsize=windowsize, n_comp=n_comp, C=C, gamma=gamma, kernel=kernel, fold=fold,
            niter=niter, oversampleopt=oversampleopt)

        return f1score, auc, testy, all_y, score_y

    def svm_permutation_test(self, bestparams, niter=100, windowsize=40, n_comp=5, kernel='rbf', fold=1,
                             oversampleopt=False, svmiter=10):

        assert isinstance(bestparams, pd.DataFrame)

        try:
            centerexit_df = self.centerexit_df
        except AttributeError:
            centerexit_df = self.get_center_exit_categories('whole', windowsize)

        rawcellsig = self.rawcellsig
        C = bestparams.C.iloc[0]
        gamma = bestparams.gamma.iloc[0]

        nbins = rawcellsig.shape[1]
        shiftidx = np.round(np.linspace(nbins / niter, nbins - nbins / niter, niter))

        f1list = []
        auclist = []

        for i in range(niter):
            print('\nIteration {} of {}'.format(i + 1, niter))
            shuffcellsig = circularly_shuffle_cellsig(rawcellsig, shiftidx[i])

            f1score, auc, __, __, __ = run_inscopix_svm(
                centerexit_df, shuffcellsig, windowsize=windowsize, n_comp=n_comp, C=C, gamma=gamma, kernel=kernel,
                fold=fold, niter=svmiter, oversampleopt=oversampleopt)

            f1list.append(f1score)
            auclist.append(auc)

        return f1list, auclist

    def run_svm_with_best_params(self, bestparams, windowsize=40, n_comp=5, kernel='rbf', fold=6, oversampleopt=False,
                                 svmiter=100):

        assert isinstance(bestparams, pd.DataFrame)

        try:
            centerexit_df = self.centerexit_df
        except AttributeError:
            centerexit_df = self.get_center_exit_categories('whole', windowsize)

        rawcellsig = self.rawcellsig
        C = bestparams['C'].iloc[0]
        gamma = bestparams['gamma'].iloc[0]

        f1score, auc, pred_y, actual_y, score_y = run_inscopix_svm(
            centerexit_df, rawcellsig, windowsize=windowsize, n_comp=n_comp, C=C, gamma=gamma, kernel=kernel,
            fold=fold, niter=svmiter, oversampleopt=oversampleopt)

        return f1score, auc, pred_y, actual_y, score_y


    def run_svm_with_diff_unit_count(self, bestparams, percunits, windowsize=40, n_comp=5, kernel='rbf', fold=6,
                                     oversampleopt=False, svmiter=20, niter=50):

        assert isinstance(bestparams, pd.DataFrame)

        # remove values smaller than 0
        try:
            centerexit_df = self.centerexit_df
        except AttributeError:
            centerexit_df = self.get_center_exit_categories('whole', windowsize)

        rawcellsig = self.rawcellsig
        totalunits = np.shape(rawcellsig)[0]
        C = bestparams.C.iloc[0]
        gamma = bestparams.gamma.iloc[0]

        svm_units = {}
        auc_score = {}

        for pc in percunits:

            print('\nCalculating SVM scores for {:.1f} of units...\n'.format(pc))

            numunits = int(np.ceil(totalunits * pc))

            if numunits < n_comp:
                warnings.warn('Number of units is smaller than number of PCA components. Returning empty list...')
                auc_score[pc] = []
                svm_units[pc] = []
                continue

            auclist = []
            units = []

            if pc == 1:
                niter = 1

            for i in range(niter):

                print('Iteration {} of {}...'.format(i+1, niter))

                unitnum = np.sort(np.random.choice(totalunits, numunits, replace=False))
                tempcellsig = rawcellsig[unitnum, :]

                __, auc, __, __, __ = run_inscopix_svm(
                    centerexit_df, tempcellsig, windowsize=windowsize, n_comp=n_comp, C=C, gamma=gamma, kernel=kernel,
                    fold=fold, niter=svmiter, oversampleopt=oversampleopt)

                auclist.append(auc)
                units.append(unitnum)

            auc_score[pc] = auclist
            svm_units[pc] = units

        return svm_units, auc_score

    def run_svm_with_diff_subgroups(self, bestparams, numunits, windowsize=40, n_comp=5, kernel='rbf', fold=6,
                                     oversampleopt=False, svmiter=20, niter=100):

        assert isinstance(bestparams, pd.Series) or isinstance(bestparams, pd.DataFrame)

        # remove values smaller than 0
        try:
            centerexit_df = self.centerexit_df
        except AttributeError:
            centerexit_df = self.get_center_exit_categories('whole', windowsize)

        rawcellsig = self.rawcellsig
        totalunits = np.shape(rawcellsig)[0]

        if isinstance(bestparams, pd.Series):
            C = bestparams.C
            gamma = bestparams.gamma
        elif isinstance(bestparams, pd.DataFrame):
            C = bestparams['C'].iloc[0]
            gamma = bestparams['gamma'].iloc[0]

        svm_units = {}
        auc_score = {}

        prefdf = self.prefdf
        categories = np.unique(prefdf['preference']).tolist()

        if isinstance(numunits, (float,int)):
            if 1 < numunits < n_comp:
                warnings.warn('Number of units is smaller than number of PCA components. Returning empty dict...')
                return svm_units, auc_score
            elif numunits > totalunits:
                warnings.warn('Number of units is larger than total number of recorded units. Returning empty dict...')
            elif numunits == totalunits:
                niter = 1
            elif 0 < numunits <= 1: # for proportions
                numunits = int(np.ceil(numunits * totalunits))
        elif isinstance(numunits, str):
            if numunits == 'all':
                numunits = np.min([np.sum(prefdf['preference'] == i) for i in categories])

        categories.append('all')

        for cat in categories:

            print('\nProcessing groups of {} cells...\n'.format(cat))
            if cat == 'all':
                catunits = list(range(np.shape(prefdf)[0]))
            else:
                catunits = prefdf.index[prefdf['preference'] == cat].tolist()
            auclist = []
            units = []

            for i in range(niter):

                if (i+1)%5 == 0:
                    print('Iteration {} of {}...'.format(i+1, niter))

                unitnum = np.sort(np.random.choice(catunits, numunits, replace=False))
                tempcellsig = rawcellsig[unitnum, :]

                __, auc, __, __, __ = run_inscopix_svm(
                    centerexit_df, tempcellsig, windowsize=windowsize, n_comp=n_comp, C=C, gamma=gamma, kernel=kernel,
                    fold=fold, niter=svmiter, oversampleopt=oversampleopt)

                auclist.append(auc)
                units.append(unitnum)

            auc_score[cat] = auclist
            svm_units[cat] = units

        return svm_units, auc_score


    def calc_center_exit_correlation(self, centerexit_df, filterthresh=400, method='epoch', plotopt=False):

        from palettable.colorbrewer.sequential import Reds_9 as R9

        cellsig = self.rawcellsig
        raster = sps.zscore(cellsig, axis=1)
        tslength = 20

        # Get all possible correlations
        closed_off = centerexit_df.loc[(centerexit_df['run'] == 'closed') & (centerexit_df['stim'] == 'off'), 'idx']
        open_off = centerexit_df.loc[(centerexit_df['run'] == 'open') & (centerexit_df['stim'] == 'off'), 'idx']
        closed_on = centerexit_df.loc[(centerexit_df['run'] == 'closed') & (centerexit_df['stim'] == 'on'), 'idx']
        open_on = centerexit_df.loc[(centerexit_df['run'] == 'open') & (centerexit_df['stim'] == 'on'), 'idx']

        if method == 'epoch':
            closed_off_raster = np.asarray([raster[:, i:i+tslength].flatten() for i in closed_off])
            open_off_raster = np.asarray([raster[:, i:i+tslength].flatten() for i in open_off])
            closed_on_raster = np.asarray([raster[:, i:i+tslength].flatten() for i in closed_on])
            open_on_raster = np.asarray([raster[:, i:i+tslength].flatten() for i in open_on])

        elif method == 'corrmat':
            closed_off_raster = np.empty((np.size(closed_off), np.size(raster, 0) ** 2))
            for i in range(np.size(closed_off)):
                temp = raster[:, closed_off.iloc[i]:closed_off.iloc[i] + tslength]
                closed_off_raster[i, :] = np.corrcoef(temp).flatten()

            open_off_raster = np.empty((np.size(open_off), np.size(raster, 0) ** 2))
            for i in range(np.size(open_off)):
                temp = raster[:, open_off.iloc[i]:open_off.iloc[i] + tslength]
                open_off_raster[i, :] = np.corrcoef(temp).flatten()

            closed_on_raster = np.empty((np.size(closed_on), np.size(raster, 0) ** 2))
            for i in range(np.size(closed_on)):
                temp = raster[:, closed_on.iloc[i]:closed_on.iloc[i] + tslength]
                closed_on_raster[i, :] = np.corrcoef(temp).flatten()

            open_on_raster = np.empty((np.size(open_on), np.size(raster, 0) ** 2))
            for i in range(np.size(open_on)):
                temp = raster[:, open_on.iloc[i]:open_on.iloc[i] + tslength]
                open_on_raster[i, :] = np.corrcoef(temp).flatten()

        # calculate off correlation matrix
        all_idx = np.concatenate((np.asarray(closed_off, dtype='int'), np.asarray(open_off, dtype='int')))
        off_corrmat = np.corrcoef(np.vstack((closed_off_raster, open_off_raster)))
        filtermat = np.zeros(np.shape(off_corrmat), dtype='bool')

        for i in range(np.size(all_idx)):
            temp = np.where(np.abs(all_idx[i] - all_idx) <= filterthresh)
            filtermat[i, temp] = True
            filtermat[temp, i] = True
        off_corrmat[filtermat] = np.nan

        if plotopt:
            plt.figure()
            plt.imshow(off_corrmat, R9.mpl_colormap)
            plt.show()

        num_closed_off = np.size(closed_off, 0)
        num_open_off = np.size(open_off, 0)
        num_total_off = num_closed_off + num_open_off

        c_c_comb = tuple(itools.combinations(range(num_closed_off), 2))
        o_o_comb = tuple(itools.combinations(range(num_closed_off, num_total_off), 2))
        c_o_comb = tuple(set(itools.combinations(range(num_total_off), 2)) - set(c_c_comb + o_o_comb))

        corrdf_off = pd.DataFrame(columns=('corrval', 'run', 'stim'))

        temp = pd.DataFrame(columns=('corrval', 'run', 'stim'))
        temp2 = off_corrmat[tuple(zip(*c_c_comb))]
        temp['corrval'] = temp2[~np.isnan(temp2)]
        temp.loc[:, ['run', 'stim']] = ['closed_closed', 'off']
        corrdf_off = pd.concat((corrdf_off, temp))

        temp = pd.DataFrame(columns=('corrval', 'run', 'stim'))
        temp2 = off_corrmat[tuple(zip(*o_o_comb))]
        temp['corrval'] = temp2[~np.isnan(temp2)]
        temp.loc[:, ['run', 'stim']] = ['open_open', 'off']
        corrdf_off = pd.concat((corrdf_off, temp))

        temp = pd.DataFrame(columns=('corrval', 'run', 'stim'))
        temp2 = off_corrmat[tuple(zip(*c_o_comb))]
        temp['corrval'] = temp2[~np.isnan(temp2)]
        temp.loc[:, ['run', 'stim']] = ['closed_open', 'off']
        corrdf_off = pd.concat((corrdf_off, temp))

        # calculate on correlation matrix
        # all_idx = np.concatenate((np.asarray(closed_on, dtype='int'), np.asarray(open_on, dtype='int')))
        # on_corrmat = np.corrcoef(np.vstack((closed_on_raster, open_on_raster)))
        # filtermat = np.zeros(np.shape(on_corrmat), dtype='bool')
        #
        # for i in range(np.size(all_idx)):
        #     temp = np.where(np.abs(all_idx[i] - all_idx) <= filterthresh)
        #     filtermat[i, temp] = True
        #     filtermat[temp, i] = True
        # on_corrmat[filtermat] = np.nan
        #
        # if plotopt:
        #     plt.figure()
        #     plt.imshow(on_corrmat, R9.mpl_colormap)
        #     plt.show()
        #
        # num_closed_on = np.size(closed_on, 0)
        # num_open_on = np.size(open_on, 0)
        # num_total_on = num_closed_on + num_open_on
        #
        # c_c_comb = tuple(itools.combinations(range(num_closed_on), 2))
        # o_o_comb = tuple(itools.combinations(range(num_closed_on, num_total_on), 2))
        # c_o_comb = tuple(set(itools.combinations(range(num_total_on), 2)) - set(c_c_comb + o_o_comb))
        #
        # corrdf_on = pd.DataFrame(columns=('corrval', 'run', 'stim'))
        #
        # temp = pd.DataFrame(columns=('corrval', 'run', 'stim'))
        # temp2 = on_corrmat[tuple(zip(*c_c_comb))]
        # temp['corrval'] = temp2[~np.isnan(temp2)]
        # temp.loc[:, ['run', 'stim']] = ['closed_closed', 'on']
        # corrdf_on = pd.concat((corrdf_on, temp))
        #
        # temp = pd.DataFrame(columns=('corrval', 'run', 'stim'))
        # temp2 = on_corrmat[tuple(zip(*o_o_comb))]
        # temp['corrval'] = temp2[~np.isnan(temp2)]
        # temp.loc[:, ['run', 'stim']] = ['open_open', 'on']
        # corrdf_on = pd.concat((corrdf_on, temp))
        #
        # temp = pd.DataFrame(columns=('corrval', 'run', 'stim'))
        # temp2 = on_corrmat[tuple(zip(*c_o_comb))]
        # temp['corrval'] = temp2[~np.isnan(temp2)]
        # temp.loc[:, ['run', 'stim']] = ['closed_open', 'on']
        # corrdf_on = pd.concat((corrdf_on, temp))

        return corrdf_off #, corrdf_on

    def bootstrap_data_to_get_desired_open_arm_time(self, percopen, niter=100, starttime=10, stdthresh=(),
                                                    prctilethresh=(), filteropt=True, split_epoch=False):

        # Ensure that only prctilethresh or stdthresh is not empty
        if not stdthresh and not prctilethresh:
            raise Exception('Enter a value for stdthresh or prctilethresh')
        elif np.isscalar(stdthresh) and np.isscalar(prctilethresh):
            raise Exception('Enter a value for either stdthresh or prctilethresh only')

        # correction for percentage instead of ratio
        if 1 < percopen <= 100:
            percopen = percopen/100

        # get anymaze time series and remove first 10 seconds (default) of time series
        anymaze_ts = self.anymaze_ts
        anymaze_ts = anymaze_ts.loc[(anymaze_ts['time'] >= starttime)]

        # get z-scored cell signal and position dataframe
        cellsig = self.rawcellsig
        posdf = self.posdf

        if split_epoch:
            boundaryidx = (0, 3600, 7200, 10800)
            numepochs = 3
            zcellsig = np.zeros(cellsig.shape)
        else:
            numepochs = 1

        if filteropt:
            from astropy.convolution import Gaussian2DKernel
            from astropy.convolution import convolve
            kernel = Gaussian2DKernel(stddev=1)

        bspref_df = []
        bs_df = []

        # get max x and y to plot calciummap/heatmap etc.
        maxxpos = np.max(anymaze_ts['xpos'])
        maxypos = np.max(anymaze_ts['ypos'])

        for j in range(numepochs):

            if split_epoch:
                print('\nProcessing epoch {}...\n'.format(j+1))
                anymaze_temp = anymaze_ts.loc[(boundaryidx[j] <= anymaze_ts.index) & (anymaze_ts.index < boundaryidx[j + 1])]
                zcellsig[:, boundaryidx[j]:boundaryidx[j+1]] = sps.zscore(cellsig[:, boundaryidx[j]:boundaryidx[j+1]], axis=1)
            else:
                anymaze_temp = anymaze_ts
                zcellsig = sps.zscore(cellsig, axis=1)

            # get open and closed arm series
            openarm = anymaze_temp.loc[(anymaze_temp['status'] == 'center') | (anymaze_temp['status'] == 'open')]
            closedarm = anymaze_temp.loc[(anymaze_temp['status'] == 'closed') | (anymaze_temp['status'] == 'approach')]

            # get number of open and closed arm pixels
            numopen = int(round(percopen * np.size(anymaze_temp,0)))
            numclosed = int(np.size(anymaze_temp,0) - numopen)

            bsprefdfall = []
            bsdfall = []

            for i in range(niter):
                print('Iteration {} of {}...'.format(i+1, niter))
                opentemp = openarm.sample(n=numopen, replace=True)
                closedtemp = closedarm.sample(n=numclosed, replace=True)
                bsdf = pd.concat((opentemp, closedtemp)).sort_index() #bootstrapped dataframe

                # plusmazemap, pos, maxxpos, maxypos, minxpos, minypos, posdf = create_plusmazemap(anymaze, startidx)

                calciummap = np.zeros((zcellsig.shape[0], maxypos + 10, maxxpos + 10))
                calciummap_thresh = np.zeros((zcellsig.shape[0], maxypos + 10, maxxpos + 10))

                heatmap = np.zeros((maxypos + 10, maxxpos + 10))

                for row in bsdf.itertuples():
                    heatmap[row.ypos, row.xpos] += 1


                for k in range(zcellsig.shape[0]):

                    for row in bsdf.itertuples():
                        calciummap[k, row.ypos, row.xpos] += zcellsig[k, row.Index]

                    norm_cmap = calciummap[k] / heatmap

                    if filteropt:
                        norm_cmap = convolve(norm_cmap, kernel=kernel, boundary=None, preserve_nan=True)

                    if np.isscalar(stdthresh):
                        cmapthresh = np.nanmean(norm_cmap) + stdthresh * np.nanstd(norm_cmap)
                    elif np.isscalar(prctilethresh):
                        cmapthresh = np.nanpercentile(norm_cmap, prctilethresh)
                    else:
                        raise Exception('Something''s wrong!')

                    cmapidx = norm_cmap >= cmapthresh
                    norm_cmap = norm_cmap * cmapidx.astype('int')
                    calciummap_thresh[k, :, :] = np.nan_to_num(norm_cmap)

                bsprefdfall.append(classify_cells_by_sig_pixels(posdf, calciummap_thresh))
                bsdfall.append(bsdf)

            bspref_df.append(bsprefdfall)
            bs_df.append(bsdfall)

        return bspref_df, bs_df


    def calc_open_vs_closed_noise_corr(self, prefopt=(), plotopt=True):

        ignorethresh = 200
        seglength = 20
        samples = 50

        zcellsig = sps.zscore(self.rawcellsig, axis=1)
        prefdf = self.prefdf

        if prefopt == 'closed':
            cellidx = prefdf[prefdf.loc[:,'preference'] == 'closed'].index
            zcellsig = zcellsig[cellidx, :]
        elif prefopt == 'open':
            cellidx = prefdf[prefdf.loc[:, 'preference'] == 'open'].index
            zcellsig = zcellsig[cellidx, :]
        elif prefopt == 'nopref':
            cellidx = prefdf[prefdf.loc[:, 'preference'] == 'nopref'].index
            zcellsig = zcellsig[cellidx, :]


        anymaze_ts = self.anymaze_ts

        closedsamples, opensamples = get_nonoverlapping_closed_open_segments(anymaze_ts, seglength, samples, ignorethresh)

        closedrate = np.empty((zcellsig.shape[0], samples))
        openrate = np.empty((zcellsig.shape[0], samples))

        for i in range(len(closedsamples)):
            idx = range(closedsamples[i], closedsamples[i]+seglength)
            cellsigtemp = zcellsig[:, idx]
            closedrate[:,i] = np.mean(cellsigtemp, axis=1)

        for i in range(len(opensamples)):

            idx = range(opensamples[i], opensamples[i]+seglength)
            cellsigtemp = zcellsig[:, idx]
            openrate[:,i] = np.mean(cellsigtemp, axis=1)

        closednoisecorr = np.corrcoef(closedrate)
        closednoisecorr = closednoisecorr[np.triu_indices(closednoisecorr.shape[0], 1)]
        opennoisecorr = np.corrcoef(openrate)
        opennoisecorr = opennoisecorr[np.triu_indices(opennoisecorr.shape[0], 1)]

        if plotopt:
            plt.figure()
            plt.scatter(closednoisecorr, opennoisecorr)
            plt.plot([0, 1], [0, 1])
            plt.xlabel('closed arms noise correlation')
            plt.ylabel('open arms noise correlation')
            plt.show()

        return closednoisecorr, opennoisecorr

    def plot_cell_perm_prefidx_vs_real_prefidx(self, permarray):

        realprefidx = self.prefdf.loc[:, 'prefidx']
        edges = np.linspace(-1, 1, 21)

        plt.figure()
        c = 1

        for i in range(realprefidx.size):
            plt.subplot(2, 3, c)

            plt.hist(permarray[:,i], bins=edges, label='Permutation PI', edgecolor='k', linewidth=1)
            plt.axvline(x=realprefidx[i], color='r', linestyle='--', label='Real PI')
            plt.ylabel('Count')
            plt.xlabel('Preference index')
            plt.title('Unit {}'.format(i))
            plt.legend()

            if c == 6:
                plt.figure()
                c = 1
            else:
                c = c + 1

    def calc_plot_prefidx_pvals(self, permprefidx, plotopt=True):

        prefdf = self.prefdf
        realpi = np.reshape(np.array(prefdf.prefidx), (1,-1))

        assert realpi.shape[1] == permprefidx.shape[1]

        compmat = np.vstack((realpi, permprefidx))
        idx = np.where(np.argsort(compmat, axis=0).T == 0)

        pvals = idx[1] / compmat.shape[0]

        if plotopt:
            plt.figure()
            binwidth = 0.05
            plt.hist(pvals, np.linspace(0,1,int(1/binwidth+1)), edgecolor='k')
            plt.xlabel('p value')
            plt.ylabel('Count')

        return pvals


def load_inscopix_anymaze_data_from_matlab(file):

    from jsbox import loadmat

    matvars = loadmat(file)
    cellsig = matvars['cell_sig']
    cellsegments = matvars['cell_segments']
    temp = matvars['anymazedata']
    fn = temp[0]._fieldnames
    anymaze = {i: np.asarray([getattr(j, i) for j in temp]) for i in fn}

    return cellsig, cellsegments, anymaze


def calculate_calciummaps(zcellsig, heatmap, pos, prctilethresh=(), stdthresh=(), filteropt=False, maxxpos=(), maxypos=()):

    assert(zcellsig.shape[1] == pos.shape[1])

    if not maxxpos:
        maxxpos = np.max(pos[1,:])
    if not maxypos:
        maxypos = np.max(pos[0,:])

    if filteropt:
        from astropy.convolution import Gaussian2DKernel
        from astropy.convolution import convolve

        kernel = Gaussian2DKernel(stddev=1)

    calciummap = np.zeros((zcellsig.shape[0], maxypos + 10, maxxpos + 10))
    calciummap_thresh = np.zeros((zcellsig.shape[0], maxypos + 10, maxxpos + 10))
    norm_cmap = np.zeros((zcellsig.shape[0], maxypos + 10, maxxpos + 10))

    for i in range(zcellsig.shape[0]):

        if not divmod(i+1, 10)[1] or i == zcellsig.shape[0] - 1:
            print('Processing cell {} of {}...'.format(i + 1, zcellsig.shape[0]))

        for j in range(zcellsig.shape[1]):
            calciummap[i, pos[0, j], pos[1, j]] += zcellsig[i, j]

        if filteropt:
            norm_cmap[i] = convolve(calciummap[i] / heatmap, kernel=kernel, boundary=None, preserve_nan=True)
        else:
            norm_cmap[i] = calciummap[i] / heatmap

        if np.isscalar(stdthresh):
            cmapthresh = np.nanmean(norm_cmap[i]) + stdthresh * np.nanstd(norm_cmap[i])
        elif np.isscalar(prctilethresh):
            cmapthresh = np.nanpercentile(norm_cmap[i], prctilethresh)
        else:
            raise Exception('Something''s wrong!')

        cmapidx = norm_cmap[i] >= cmapthresh
        tempmap = norm_cmap[i] * cmapidx.astype('int')
        calciummap_thresh[i, :, :] = np.nan_to_num(tempmap)

    return calciummap, calciummap_thresh, norm_cmap


def create_plusmazemap(anymaze, startidx=0):

    # replace all empty lists with nans (empty lists were causing lots of bugs)
    anymaze.mask(anymaze.applymap(str).eq('[]'), inplace=True)
    # replace all nans with the first valid position
    anymaze.CentrePosnX.fillna(anymaze.CentrePosnX[anymaze.CentrePosnX.first_valid_index()], inplace=True)
    anymaze.CentrePosnY.fillna(anymaze.CentrePosnY[anymaze.CentrePosnY.first_valid_index()], inplace=True)
    # convert x- and y-positions to integers for indexing later
    anymaze.CentrePosnX = anymaze.CentrePosnX.astype(int)
    anymaze.CentrePosnY = anymaze.CentrePosnY.astype(int)

    xpos = anymaze['CentrePosnX']
    ypos = anymaze['CentrePosnY']

    # # Replace empty arrays with first position
    # nonzeroidx = np.squeeze(np.nonzero(xpos))
    # xpos[0:nonzeroidx[0]] = xpos[nonzeroidx[0]]
    # ypos[0:nonzeroidx[0]] = ypos[nonzeroidx[0]]

    # Create heat map area for each cell
    minxpos = np.min(xpos)
    minypos = np.min(ypos)

    xpos = xpos - minxpos + 10
    ypos = ypos - minypos + 10

    maxxpos = np.max(xpos)
    maxypos = np.max(ypos)

    pos = np.stack((ypos, xpos))

    posdf = pd.DataFrame(np.unique(pos[:,startidx:], axis=1).T, columns=['y','x'])
    posdf.insert(2, 'posclass', np.nan)
    plusmazemap = np.zeros((maxypos + 10, maxxpos + 10))

    # Update anymaze with new positions
    anymaze['CentrePosnX'] = anymaze['CentrePosnX'] - minxpos + 10
    anymaze['CentrePosnY'] = anymaze['CentrePosnY'] - minypos + 10

    # Populate unique positions (posdf) and where they are in the plusmaze
    for i in range(posdf.shape[0]):

        idx = np.where((np.asarray(posdf.loc[i, ['y','x']]) == pos.T).all(axis=1))[0][0]

        if anymaze.at[idx, 'InApproach']:
            posdf.loc[i, 'posclass'] = 'approach'
            plusmazemap[posdf.loc[i,'y'], posdf.loc[i,'x']] = 1
        elif anymaze.at[idx, 'InClosedArms']:
            posdf.loc[i, 'posclass'] = 'closed'
            plusmazemap[posdf.loc[i, 'y'], posdf.loc[i, 'x']] = 2
        elif anymaze.at[idx, 'InOpenArms']:
            posdf.loc[i, 'posclass'] = 'open'
            plusmazemap[posdf.loc[i, 'y'], posdf.loc[i, 'x']] = 3
        elif anymaze.at[idx, 'InCenter']:
            posdf.loc[i, 'posclass'] = 'center'
            plusmazemap[posdf.loc[i, 'y'], posdf.loc[i, 'x']] = 4
        else:
            posdf.loc[i, 'posclass'] = 'bad'

    if (posdf.posclass == 'bad').any():
        warnings.warn('There are indices with no category')


    return anymaze, plusmazemap, posdf, pos, maxxpos, maxypos


def plot_average_categorical_signal(calciummap, idxdict):

    idxdict.pop('leftovers', None)

    caaveragemap = []

    plt.figure()
    c = 1

    for i in range(calciummap.shape[0]):

        temp = dict.fromkeys(idxdict.keys(), [])
        plt.subplot(2, 3, c)

        for key, value in idxdict.items():

            temp[key] = [calciummap[i, j[0], j[1]] for j in idxdict[key] if calciummap[i, j[0], j[1]] != 0]

        # closedarms = temp['closed'] + temp['approach']
        # openarms = temp['center'] + temp['open']

        closedratio = len(temp['closed'])/len(idxdict['closed'])
        approachratio = len(temp['approach'])/len(idxdict['approach'])
        openratio = len(temp['open'])/len(idxdict['open'])
        centerratio = len(temp['center'])/len(idxdict['center'])
        allratio = (closedratio, approachratio, openratio, centerratio)

        plt.plot((0,1,2,3), allratio)


            # temp[key] = temp[key] / len(idxdict[key])

        caaveragemap.append(temp)

        if c == 6:
            plt.figure()
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            c = 1
        else:
            c = c + 1


def classify_cells_by_sig_pixels(posdf, calciummap_thresh, pval=0.05):

    # Closed: closed arms + in approach, Open: open arms + center
    closed_idx = pd.concat((posdf.loc[posdf.posclass == 'closed', ['y','x']],
                                posdf.loc[posdf.posclass == 'approach', ['y','x']]))
    open_idx = pd.concat((posdf.loc[posdf.posclass == 'open', ['y','x']],
                                posdf.loc[posdf.posclass == 'center', ['y','x']]))

    # closed_idx = posdf['closed']
    # open_idx = np.concatenate((posdf['open'], posdf['center'], posdf['approach']), axis=0)

    # Number of closed and open pixels
    closed_total = np.size(closed_idx, axis=0)
    open_total = np.size(open_idx, axis=0)

    # Initialize arrays
    closed_sig_ratio = np.zeros(np.size(calciummap_thresh,0))
    open_sig_ratio = np.zeros(np.size(calciummap_thresh,0))

    # prefdf = pd.DataFrame(index=list(range(np.size(self.rawcellsig,0))), columns=('pvals','preference','prefidx'))
    prefdf = pd.DataFrame(index=(list(range(np.size(calciummap_thresh,0)))), columns=('pvals','preference','prefidx'))


    for i in range(np.size(calciummap_thresh,0)):
        idx = np.squeeze(np.where(calciummap_thresh[i] > 0)).T

        # find closed and open significant pixels for each unit
        closed_sig = np.squeeze([j for j in idx if np.any((j == closed_idx).all(axis=1))])
        open_sig = np.squeeze([j for j in idx if np.any((j == open_idx).all(axis=1))])
        closed_sig_num = np.size(closed_sig,0)
        open_sig_num = np.size(open_sig,0)

        # get expected values for chi-squared test
        expectedratio = (closed_sig_num + open_sig_num) / (closed_total + open_total)
        closed_sig_ratio[i] = closed_sig_num/closed_total
        open_sig_ratio[i] = open_sig_num/open_total
        expectedopen = np.round(expectedratio * open_total)
        expectedclosed = np.round(expectedratio * closed_total)
        expectedvec = [expectedclosed, expectedopen, closed_total - expectedclosed, open_total - expectedopen]
        observedvec = [closed_sig_num, open_sig_num, closed_total - closed_sig_num, open_total - open_sig_num]
        prefdf.loc[i, 'pvals'] = sps.chisquare(observedvec, expectedvec)[1] # chi-squared test

    # get significant and non-significant cells
    nonsigpvals = np.squeeze(np.where(prefdf.pvals > pval))

    if nonsigpvals.size > 1:
        prefdf.loc[nonsigpvals, 'preference'] = 'nopref'
    elif nonsigpvals.size == 1:
        prefdf.loc[int(nonsigpvals), 'preference'] = 'nopref'

    sigpvals = np.squeeze(np.where(prefdf.pvals <= pval))

    # separate significant cells into open-preferring and close-preferring
    if sigpvals.size > 1:

        for i in sigpvals:
            if open_sig_ratio[i] > closed_sig_ratio[i]:
                prefdf.loc[i, 'preference'] = 'open'
            else:
                prefdf.loc[i, 'preference'] = 'closed'

    elif sigpvals.size == 1:
        sigpvals = int(sigpvals)
        if open_sig_ratio[sigpvals] > closed_sig_ratio[sigpvals]:
            prefdf.loc[sigpvals, 'preference'] = 'open'
        else:
            prefdf.loc[sigpvals, 'preference'] = 'closed'



    # get preference index (-1 to 1)
    prefdf.loc[:, 'prefidx'] = (open_sig_ratio - closed_sig_ratio) / (open_sig_ratio + closed_sig_ratio)

    return prefdf

def plot_epochs_prefidx_cdf(prefidxdf):

    plt.figure()

    allmat = np.array(np.stack(pf.prefidx for pf in prefidxdf))

    from statsmodels.distributions.empirical_distribution import ECDF

    for i in range(allmat.shape[0]):
        ecdf = ECDF(allmat[i,:])
        plt.plot(ecdf.x, ecdf.y)

    plt.legend(('off1','on','off2'))
    plt.xlabel('Preference index')
    plt.ylabel('Cumulative freq')


def plot_prefidx_subsets(prefidxdf):

    plt.figure()

    allmat = np.array(np.stack(pf.prefidx for pf in prefidxdf))
    for i in range(np.size(allmat,1)):
        plt.plot(range(np.size(allmat,0)), allmat[:,i], Color=[.7,.7,.7], alpha=0.5, linewidth=0.5)

    plt.errorbar(range(np.size(allmat,0)), np.mean(allmat,axis=1), yerr=sps.sem(allmat,axis=1), Color='k', ecolor='k', zorder=3)
    # plt.xlim((-.5, 2.5))
    # plt.xticks(np.arange(3), ('Off', 'On', 'Off'))
    plt.ylabel('Preference index')
    plt.title('All')

    comb = list(itools.combinations(range(3), 2))
    pvals = [sps.ttest_rel(allmat[c[0], :], allmat[c[1], :])[1] for c in comb]
    plt.show()

    return np.hstack((np.array(comb), np.reshape(pvals, (3,1))))

    # incidx = [x[0] <= x[1] for x in list(zip(prefidxoff1, prefidxon))]
    # plt.subplot(232)
    # incidxmat = np.stack((prefidxoff1[incidx], prefidxon[incidx], prefidxoff2[incidx]))
    # for i in range(np.size(incidxmat, 1)):
    #     plt.plot(range(np.size(incidxmat, 0)), incidxmat[:, i], Color=[.7, .7, .7], alpha=0.5)
    #
    # plt.errorbar(range(np.size(incidxmat, 0)), np.mean(incidxmat, axis=1), yerr=sps.sem(incidxmat, axis=1),
    #              Color='k', ecolor='k', zorder=3)
    # plt.xlim((-.5, 2.5))
    # plt.xticks(np.arange(3), ('Off', 'On', 'Off'))
    # plt.ylabel('Preference index')
    # plt.title('Off 1 < On')
    #
    #
    # decidx = [x[0] > x[1] for x in list(zip(prefidxoff1, prefidxon))]
    # plt.subplot(233)
    # decidxmat = np.stack((prefidxoff1[decidx], prefidxon[decidx], prefidxoff2[decidx]))
    # for i in range(np.size(decidxmat, 1)):
    #     plt.plot(range(np.size(decidxmat, 0)), decidxmat[:, i], Color=[.7, .7, .7], alpha=0.5)
    #
    # plt.errorbar(range(np.size(decidxmat, 0)), np.mean(decidxmat, axis=1), yerr=sps.sem(decidxmat, axis=1),
    #              Color='k', ecolor='k', zorder=3)
    # plt.xlim((-.5, 2.5))
    # plt.xticks(np.arange(3), ('Off', 'On', 'Off'))
    # plt.ylabel('Preference index')
    # plt.title('Off 1 > On')
    #
    # nopref = [-0.25 <= i <= 0.25 for i in prefidxoff1]
    # plt.subplot(234)
    # noprefmat = np.stack((prefidxoff1[nopref], prefidxon[nopref], prefidxoff2[nopref]))
    # for i in range(np.size(noprefmat, 1)):
    #     plt.plot(range(np.size(noprefmat, 0)), noprefmat[:, i], Color=[.7, .7, .7], alpha=0.5)
    #
    # plt.errorbar(range(np.size(noprefmat, 0)), np.mean(noprefmat, axis=1), yerr=sps.sem(noprefmat, axis=1), Color='k',
    #              ecolor='k', zorder=3)
    # plt.xlim((-.5, 2.5))
    # plt.xticks(np.arange(3), ('Off', 'On', 'Off'))
    # plt.ylabel('Preference index')
    # plt.title('-0.25 <= Off 1 <= 0.25')
    #
    # openpref = [i > 0.25 for i in prefidxoff1]
    # plt.subplot(235)
    # openprefmat = np.stack((prefidxoff1[openpref], prefidxon[openpref], prefidxoff2[openpref]))
    # for i in range(np.size(openprefmat, 1)):
    #     plt.plot(range(np.size(openprefmat, 0)), openprefmat[:, i], Color=[.7, .7, .7], alpha=0.5)
    #
    # plt.errorbar(range(np.size(openprefmat, 0)), np.mean(openprefmat, axis=1), yerr=sps.sem(openprefmat, axis=1),
    #              Color='k', ecolor='k', zorder=3)
    # plt.xlim((-.5, 2.5))
    # plt.xticks(np.arange(3), ('Off', 'On', 'Off'))
    # plt.ylabel('Preference index')
    # plt.title('Off 1 > 0.25')
    #
    # closedpref = [i < -0.25 for i in prefidxoff1]
    # plt.subplot(236)
    # closedprefmat = np.stack((prefidxoff1[closedpref], prefidxon[closedpref], prefidxoff2[closedpref]))
    # for i in range(np.size(closedprefmat, 1)):
    #     plt.plot(range(np.size(closedprefmat, 0)), closedprefmat[:, i], Color=[.7, .7, .7], alpha=0.5)
    #
    # plt.errorbar(range(np.size(closedprefmat, 0)), np.mean(closedprefmat, axis=1), yerr=sps.sem(closedprefmat, axis=1),
    #              Color='k', ecolor='k', zorder=3)
    # plt.xlim((-.5, 2.5))
    # plt.xticks(np.arange(3), ('Off', 'On', 'Off'))
    # plt.ylabel('Preference index')
    # plt.title('Off 1 < -0.25')
    #


# def plot_prefidx_subsets(prefidxdf):
#
#     plt.figure()
#
#     # prefidxoff1 = prefidxdf[0].prefidx
#     # prefidxon = prefidxdf[1].prefidx
#     # prefidxoff2 = prefidxdf[2].prefidx
#
#     allmat = np.array(np.stack(pf.prefidx for pf in prefidxdf))
#     # plt.subplot(231)
#     for i in range(np.size(allmat,1)):
#         plt.plot(range(np.size(allmat,0)), allmat[:,i], Color=[.7,.7,.7], alpha=0.5)
#
#     plt.errorbar(range(np.size(allmat,0)), np.mean(allmat,axis=1), yerr=sps.sem(allmat,axis=1), Color='k', ecolor='k', zorder=3)
#     plt.xlim((-.5, 2.5))
#     plt.xticks(np.arange(3), ('Off', 'On', 'Off'))
#     plt.ylabel('Preference index')
#     plt.title('All')
#
#
#     plt.show()




def convert_cellsig_to_raster(cellsig, prctilethresh=95, timethresh=(), binsize=50):

    cellsigrast = np.empty((np.shape(cellsig)))

    for i in range(np.size(cellsig,0)):
        sigthresh = np.percentile(cellsig[i], prctilethresh)
        raster = (cellsig[i] >= sigthresh).astype('int')

        if timethresh:
            rleraster = jsbox.rude(raster) # run-length encoding of raster

            for j in range(len(rleraster[0])):
                if rleraster[0,j] == 1 and rleraster[1,j] <  timethresh/binsize:
                    rleraster[0,j] = 0 # remove short epochs of 'activity' set by time-threshold (usually 500ms)

            cellsigrast[i,:] = jsbox.rude(rleraster)

        else:
            cellsigrast[i,:] = raster

    return cellsigrast

def process_anymaze_timeseries(anymaze, simplifyopt=0):
    """
    Function to convert anymaze into time series for the x-y and categorical positions of mouse
    :param anymaze: raw anymaze dictionary
    :param simplifyopt: If 0 (default), parses if mouse is in approach, center, open or closed. If 1, parses only
    center, open or closed.
    :return: Time series with time, x-y position and categorical position
    """

    anymaze_ts = pd.DataFrame(columns=('time','ypos','xpos','status'))
    anymaze_ts.loc[:,'time'] = anymaze['Time']
    anymaze_ts.loc[:,'ypos'] = anymaze['CentrePosnY']
    anymaze_ts.loc[:,'xpos'] = anymaze['CentrePosnX']

    if simplifyopt == 0:
        for i in range(len(anymaze['Time'])):
            if anymaze['InApproach'][i] == 1:
                anymaze_ts.loc[i,'status'] = 'approach'
            elif anymaze['InCenter'][i] == 1:
                anymaze_ts.loc[i,'status'] = 'center'
            elif anymaze['InClosedArms'][i] == 1:
                anymaze_ts.loc[i,'status'] = 'closed'
            elif anymaze['InOpenArms'][i] == 1:
                anymaze_ts.loc[i,'status'] = 'open'
            else:
                warnings.warn('Index {} has no category'.format(i))
    else:
        for i in range(len(anymaze['Time'])):
            if anymaze['InCenter'][i] == 1:
                anymaze_ts.loc[i,'status'] = 'center'
            elif anymaze['InClosedArms'][i] == 1:
                anymaze_ts.loc[i,'status'] = 'closed'
            elif anymaze['InOpenArms'][i] == 1:
                anymaze_ts.loc[i,'status'] = 'open'
            else:
                warnings.warn('Index {} has no category'.format(i))


    return anymaze_ts

def calc_epoch_prefidx_corrcoef(prefidxdf):

    prefidxoff1 = np.asarray(prefidxdf[0].prefidx)
    prefidxon = np.asarray(prefidxdf[1].prefidx)
    prefidxoff2 = np.asarray(prefidxdf[2].prefidx)

    temp = np.stack((prefidxoff1, prefidxon, prefidxoff2))

    return np.corrcoef(temp)

def calc_mouse_speed_from_anymaze_data(anymaze_ts, sigma=5):

    import scipy.ndimage.filters as filt
    xpos = np.asarray(anymaze_ts.xpos)
    ypos = np.asarray(anymaze_ts.ypos)
    speed = np.concatenate([[0], np.sqrt(np.diff(xpos) ** 2 + np.diff(ypos) ** 2)])
    speed = filt.gaussian_filter1d(speed, sigma)
    anymaze_ts['speed'] = speed

    return anymaze_ts

def apply_2d_gauss_filter_to_cellsigmap(cellsigmaze):

    from astropy.convolution import Gaussian2DKernel
    from astropy.convolution import convolve

    kernel = Gaussian2DKernel(stddev=2)
    return convolve(cellsigmaze, kernel)


def plot_perc_pref_open_vs_perc_spent_in_open(mousedict):

    percopen = []
    percprefopen = []
    percprefclosed = []
    medpi = []
    meanpi = []

    for i in mousedict.values():

        pref = i.prefdf['preference']
        posstatus = i.anymaze_ts['status']
        pi = i.prefdf['prefidx']

        percopen.append(np.size(posstatus.loc[(posstatus=='open') | (posstatus=='center')]) / np.size(posstatus))
        percprefopen.append(np.size(pref.loc[pref == 'open']) / np.size(pref))
        percprefclosed.append(np.size(pref.loc[pref == 'closed']) / np.size(pref))
        medpi.append(np.median(pi))
        meanpi.append(np.mean(pi))

    logpercopen = np.log10(percopen)

    plt.figure()
    plt.scatter(logpercopen, percprefopen)
    slope, intercept, r_value, p_value, __ = sps.linregress(logpercopen, percprefopen)
    xi = np.linspace(np.min(logpercopen), np.max(logpercopen))
    yi = xi * slope + intercept
    plt.plot(xi,yi,'r--')
    xl = plt.xlim()
    yl = plt.ylim()
    plt.text(xl[0]+(xl[1]-xl[0])*.75, yl[0]+(yl[1]-yl[0])*.25,
             '$r^2$ = {:.3f}\np = {:.3f}'.format(r_value**2, p_value))
    plt.xlabel('Percentage time in open arm')
    plt.ylabel('Percentage of open-arm preferring cells')

    plt.figure()
    plt.scatter(np.log10(percopen), percprefclosed)
    plt.xlabel('Percentage time in open arm')
    plt.ylabel('Percentage of closed-arm preferring cells')

    plt.figure()
    plt.scatter(logpercopen, medpi)
    slope, intercept, r_value, p_value, __ = sps.linregress(logpercopen, medpi)
    xi = np.linspace(np.min(logpercopen), np.max(logpercopen))
    yi = xi * slope + intercept
    plt.plot(xi, yi, 'r--')
    xl = plt.xlim()
    yl = plt.ylim()
    plt.text(xl[0] + (xl[1] - xl[0]) * .75, yl[0] + (yl[1] - yl[0]) * .25,
             '$r^2$ = {:.3f}\np = {:.3f}'.format(r_value ** 2, p_value))
    plt.xlabel('Percentage time in open arm')
    plt.ylabel('Median preference index')

    plt.figure()
    plt.scatter(logpercopen, meanpi)
    slope, intercept, r_value, p_value, __ = sps.linregress(logpercopen, meanpi)
    xi = np.linspace(np.min(logpercopen), np.max(logpercopen))
    yi = xi * slope + intercept
    plt.plot(xi, yi, 'r--')
    xl = plt.xlim()
    yl = plt.ylim()
    plt.text(xl[0] + (xl[1] - xl[0]) * .75, yl[0] + (yl[1] - yl[0]) * .25,
             '$r^2$ = {:.3f}\np = {:.3f}'.format(r_value ** 2, p_value))
    plt.xlabel('Percentage time in open arm')
    plt.ylabel('Mean preference index')

    plt.figure()
    plt.scatter(percopen, medpi)
    plt.xlabel('Percentage time in open arm')
    plt.ylabel('Median preference index')


def plot_perc_pref_vs_perc_time_spent_by_epoch(mousetuple, mouseepochdf):

    assert(len(mousetuple) == len(mouseepochdf))
    numepoch = 3
    boundaryidx = (200, 3600, 7200, 10800)

    percopen = []
    percprefopen = []
    medpi = []

    for i in range(len(mousetuple)):

        for j in range(numepoch):

            posstatus = mousetuple[i].anymaze_ts['status'][boundaryidx[j]:boundaryidx[j+1]]
            pref = mouseepochdf[i][j]['preference']
            pi = mouseepochdf[i][j]['prefidx']

            percopen.append(np.size(posstatus.loc[(posstatus == 'open') | (posstatus == 'center')]) / np.size(posstatus))
            percprefopen.append(np.size(pref.loc[pref == 'open']) / np.size(pref))
            medpi.append(np.median(pi))

    L = np.size(percopen)

    plt.figure()
    plt.scatter(percopen[0:L:numepoch], percprefopen[0:L:numepoch])
    plt.scatter(percopen[1:L:numepoch], percprefopen[1:L:numepoch])
    plt.scatter(percopen[2:L:numepoch], percprefopen[2:L:numepoch])
    plt.legend(('epoch 1', 'epoch 2', 'epoch 3'))
    plt.xlabel('Percentage time in open arms')
    plt.ylabel('Percentage of open-arm preferring cells')

    plt.figure()
    plt.scatter(percopen[0:L:numepoch], medpi[0:L:numepoch])
    plt.scatter(percopen[1:L:numepoch], medpi[1:L:numepoch])
    plt.scatter(percopen[2:L:numepoch], medpi[2:L:numepoch])
    plt.legend(('epoch 1', 'epoch 2', 'epoch 3'))
    plt.xlabel('Percentage time in open arm')
    plt.ylabel('Median preference index')

    plt.figure()
    for i in range(len(mousetuple)):
        plt.scatter(percopen[i*numepoch:(i+1)*numepoch], percprefopen[i*numepoch:(i+1)*numepoch])
    plt.legend(('m148', 'm149', 'm325', 'm2049', 'm2225', 'm2383'))
    plt.xlabel('Percentage time in open arms')
    plt.ylabel('Percentage of open-arm preferring cells')

    plt.figure()
    for i in range(len(mousetuple)):
        plt.scatter(percopen[i*numepoch:(i+1)*numepoch], medpi[i*numepoch:(i+1)*numepoch])
    plt.legend(('m148', 'm149', 'm325', 'm2049', 'm2225', 'm2383'))
    plt.xlabel('Percentage time in open arm')
    plt.ylabel('Median preference index')


def plot_pref_idx_ecdf(mousedict):

    from statsmodels.distributions.empirical_distribution import ECDF
    plt.figure()
    legendlab = []

    for i in mousedict.values():
        ecdf = ECDF(i.prefdf['prefidx'])
        plt.plot(ecdf.x, ecdf.y)
        legendlab.append(i.mouse)

    plt.legend(legendlab)
    plt.xlabel('Preference index')
    plt.ylabel('Cumulative freq')


def batch_load_bootstrapped_data(folder=r'I:\Inscopix_Data\Bootstrapped', perc=30):

    files = glob.glob(os.path.join(folder,'*bootstrap_{:2.1f}*.pickle'.format(perc)))

    bsdict = {}

    for i in files:

        with open(i, 'rb') as f:
            temp = pickle.load(f)

        mousenum = re.search('m\d{3,4}(?=_RTEPM)', i).group(0)
        bsdict[mousenum] = temp[mousenum + '_bspref']

    return bsdict

def batch_pickle2dict(folder, iden=()):

    if not iden:
        iden = 'm*.pickle'

    files = glob.glob(os.path.join(folder, iden))

    pickdict = {}

    for i in files:
        with open(i, 'rb') as f:
            temp = pickle.load(f)

        mousenum = re.search('m\d{3,4}', i).group(0)
        pickdict[mousenum] = temp

    return pickdict


def plot_permutation_cell_pref(mdict, permdict):

    assert(np.all(mdict.keys() == permdict.keys()))

    perctimeopen = np.empty(len(mdict.keys()))
    medpi = np.empty((len(mdict.keys()), 100))
    mnpi = np.empty((len(mdict.keys()), 100))
    realmnpi = np.empty((len(mdict.keys()),1))
    realmedpi = np.empty((len(mdict.keys()),1))

    for c,i,j in zip(itools.count(), mdict.values(), permdict.values()):

        perctimeopen[c] = i.percopen
        realmnpi[c] = np.mean(i.prefdf['prefidx'])
        realmedpi[c] = np.median(i.prefdf['prefidx'])

        medpi[c,:] = np.median(j, axis=1)
        mnpi[c,:] = np.mean(j, axis=1)

    mean_medpi = np.mean(medpi, axis=1)
    sem_medpi = sps.sem(medpi, axis=1)

    mean_mnpi = np.mean(mnpi, axis=1)
    sem_mnpi = sps.sem(mnpi, axis=1)

    sub_mean_medpi = np.mean(realmedpi - medpi, axis=1)
    sub_sem_medpi = sps.sem(realmedpi - medpi, axis=1)

    sub_mean_mnpi = np.mean(realmnpi - mnpi, axis=1)
    sub_sem_mnpi = sps.sem(realmnpi - mnpi, axis=1)


    logpercopen = np.log10(perctimeopen)

    xi = np.linspace(np.min(logpercopen), np.max(logpercopen))
    yi = lambda y: regres.slope * y + regres.intercept

    plt.figure()
    plt.errorbar(logpercopen, mean_medpi, yerr=sem_medpi, fmt='o')
    regres = sps.linregress(logpercopen, mean_medpi)
    plt.plot(xi, yi(xi), 'r--')
    xl = plt.xlim()
    yl = plt.ylim()
    plt.text(xl[0] + (xl[1] - xl[0]) * .75, yl[0] + (yl[1] - yl[0]) * .25,
             '$r^2$ = {:.3f}\np = {:.3f}'.format(regres.rvalue ** 2, regres.pvalue))
    plt.xlabel('Percentage time spent in open arm,\nREAL data')
    plt.ylabel('Mean of median preference index,\nPERMUTATION data')

    plt.figure()
    plt.errorbar(logpercopen, mean_mnpi, yerr=sem_mnpi, fmt='o')
    regres = sps.linregress(logpercopen, mean_mnpi)
    plt.plot(xi, yi(xi), 'r--')
    xl = plt.xlim()
    yl = plt.ylim()
    plt.text(xl[0] + (xl[1] - xl[0]) * .75, yl[0] + (yl[1] - yl[0]) * .25,
             '$r^2$ = {:.3f}\np = {:.3f}'.format(regres.rvalue ** 2, regres.pvalue))
    plt.xlabel('Percentage time spent in open arm,\nREAL data')
    plt.ylabel('Mean of mean preference index,\nPERMUTATION data')

    plt.figure()
    plt.errorbar(logpercopen, sub_mean_medpi, yerr=sub_sem_medpi, fmt='o')
    regres = sps.linregress(logpercopen, sub_mean_medpi)
    plt.plot(xi, yi(xi), 'r--')
    xl = plt.xlim()
    yl = plt.ylim()
    plt.text(xl[0] + (xl[1] - xl[0]) * .75, yl[0] + (yl[1] - yl[0]) * .25,
             '$r^2$ = {:.3f}\np = {:.3f}'.format(regres.rvalue ** 2, regres.pvalue))
    plt.xlabel('Percentage time spent in open arm,\nREAL data')
    plt.ylabel('Real median preference index - mean of median preference index,\nPERMUTATION data')

    plt.figure()
    plt.errorbar(logpercopen, sub_mean_mnpi, yerr=sub_sem_mnpi, fmt='o')
    regres = sps.linregress(logpercopen, sub_mean_mnpi)
    plt.plot(xi, yi(xi), 'r--')
    xl = plt.xlim()
    yl = plt.ylim()
    plt.text(xl[0] + (xl[1] - xl[0]) * .75, yl[0] + (yl[1] - yl[0]) * .25,
             '$r^2$ = {:.3f}\np = {:.3f}'.format(regres.rvalue ** 2, regres.pvalue))
    plt.xlabel('Percentage time spent in open arm,\nREAL data')
    plt.ylabel('Real mean preference index - mean of mean preference index,\nPERMUTATION data')


def plot_bootstrapped_cell_pref(mousedict, bsdict):

    assert(np.all(mousedict.keys() == bsdict.keys()))

    perctimeopen = np.empty(len(mousedict.keys()))
    mean_percopenpref = np.empty(len(mousedict.keys()))
    sem_percopenpref = np.empty(len(mousedict.keys()))
    mean_percclosedpref = np.empty(len(mousedict.keys()))
    sem_percclosedpref = np.empty(len(mousedict.keys()))
    mean_medpi = np.empty(len(mousedict.keys()))
    sem_medpi = np.empty(len(mousedict.keys()))
    mean_mnpi = np.empty(len(mousedict.keys()))
    sem_mnpi = np.empty(len(mousedict.keys()))


    for c,(i,j) in enumerate(zip(mousedict.values(), bsdict.values())):

        perctimeopen[c] = i.percopen
        prefdf = j[0]

        temp_openpref = [k.loc[k['preference'] == 'open'].shape[0]/k.shape[0] for k in prefdf]
        mean_percopenpref[c] = np.mean(temp_openpref)
        sem_percopenpref[c] = sps.sem(temp_openpref)

        temp_closedpref = [k.loc[k['preference'] == 'closed'].shape[0]/k.shape[0] for k in prefdf]
        mean_percclosedpref[c] = np.mean(temp_closedpref)
        sem_percclosedpref[c] = sps.sem(temp_closedpref)

        temp_pi = [np.median(k.prefidx) for k in prefdf]
        mean_medpi[c] = np.mean(temp_pi)
        sem_medpi[c] = sps.sem(temp_pi)

        temp_pi = [np.mean(k.prefidx) for k in prefdf]
        mean_mnpi[c] = np.mean(temp_pi)
        sem_mnpi[c] = sps.sem(temp_pi)


    logpercopen = np.log10(perctimeopen)

    plt.figure()
    plt.errorbar(logpercopen, mean_percopenpref, yerr=sem_percopenpref, fmt='o')
    regres = sps.linregress(logpercopen, mean_percopenpref)
    xi = np.linspace(np.min(logpercopen), np.max(logpercopen))
    yi = lambda y: regres.slope*y + regres.intercept
    plt.plot(xi, yi(xi), 'r--')
    xl = plt.xlim()
    yl = plt.ylim()
    plt.text(xl[0] + (xl[1] - xl[0]) * .75, yl[0] + (yl[1] - yl[0]) * .25,
             '$r^2$ = {:.3f}\np = {:.3f}'.format(regres.rvalue ** 2, regres.pvalue))
    plt.xlabel('Percentage time spent in open arm,\nREAL data')
    plt.ylabel('Mean percentage of open-preferring cells,\n100 iterations, BOOTSTRAPPED data')

    plt.figure()
    plt.errorbar(logpercopen, mean_percclosedpref, yerr=sem_percclosedpref, fmt='o')
    regres = sps.linregress(logpercopen, mean_percclosedpref)
    plt.plot(xi, yi(xi), 'r--')
    xl = plt.xlim()
    yl = plt.ylim()
    plt.text(xl[0] + (xl[1] - xl[0]) * .75, yl[0] + (yl[1] - yl[0]) * .25,
             '$r^2$ = {:.3f}\np = {:.3f}'.format(regres.rvalue ** 2, regres.pvalue))
    plt.xlabel('Percentage time spent in open arm,\nREAL data')
    plt.ylabel('Mean percentage of closed-preferring cells,\n100 iterations, BOOTSTRAPPED data')

    plt.figure()
    plt.errorbar(logpercopen, mean_medpi, yerr=sem_medpi, fmt='o')
    regres = sps.linregress(logpercopen, mean_medpi)
    plt.plot(xi, yi(xi), 'r--')
    xl = plt.xlim()
    yl = plt.ylim()
    plt.text(xl[0] + (xl[1] - xl[0]) * .75, yl[0] + (yl[1] - yl[0]) * .25,
             '$r^2$ = {:.3f}\np = {:.3f}'.format(regres.rvalue ** 2, regres.pvalue))
    plt.xlabel('Percentage time spent in open arm,\nREAL data')
    plt.ylabel('Mean of median preference index,\n100 iterations, BOOTSTRAPPED data')

    plt.figure()
    plt.errorbar(logpercopen, mean_mnpi, yerr=sem_mnpi, fmt='o')
    regres = sps.linregress(logpercopen, mean_mnpi)
    plt.plot(xi, yi(xi), 'r--')
    xl = plt.xlim()
    yl = plt.ylim()
    plt.text(xl[0] + (xl[1] - xl[0]) * .75, yl[0] + (yl[1] - yl[0]) * .25,
             '$r^2$ = {:.3f}\np = {:.3f}'.format(regres.rvalue ** 2, regres.pvalue))
    plt.xlabel('Percentage time spent in open arm,\nREAL data')
    plt.ylabel('Mean of mean preference index,\n100 iterations, BOOTSTRAPPED data')


def plot_bootstrapped_cell_pref_split_epoch(mousetuple, bootstraptuple):

    assert (len(mousetuple) == len(bootstraptuple))

    numepoch = 3

    perctimeopen = np.empty(len(mousetuple)*numepoch)
    mean_percopenpref = np.empty(len(mousetuple)*numepoch)
    sem_percopenpref = np.empty(len(mousetuple)*numepoch)
    mean_percclosedpref = np.empty(len(mousetuple)*numepoch)
    sem_percclosedpref = np.empty(len(mousetuple)*numepoch)
    mean_medpi = np.empty(len(mousetuple)*numepoch)
    sem_medpi = np.empty(len(mousetuple)*numepoch)

    for i in range(len(mousetuple)):

        for k in range(len(bootstraptuple[i])):

            if k == 0:
                ts = mousetuple[i].anymaze_ts.iloc[200:3600]
            elif k == 1:
                ts = mousetuple[i].anymaze_ts.iloc[3600:7200]
            else:
                ts = mousetuple[i].anymaze_ts.iloc[7200:10800]

            perctimeopen[i*numepoch+k] = ts.loc[(ts['status'] == 'center') | (ts['status'] == 'open')].shape[0] / ts.shape[0]
            prefdf = bootstraptuple[i][k]

            temp_openpref = [j.loc[j['preference'] == 'open'].shape[0] / j.shape[0] for j in prefdf]
            mean_percopenpref[i*numepoch+k] = np.mean(temp_openpref)
            sem_percopenpref[i*numepoch+k] = sps.sem(temp_openpref)

            temp_closedpref = [j.loc[j['preference'] == 'closed'].shape[0] / j.shape[0] for j in prefdf]
            mean_percclosedpref[i*numepoch+k] = np.mean(temp_closedpref)
            sem_percclosedpref[i*numepoch+k] = sps.sem(temp_closedpref)

            temp_pi = [np.median(j.prefidx) for j in prefdf]
            mean_medpi[i*numepoch+k] = np.mean(temp_pi)
            sem_medpi[i*numepoch+k] = sps.sem(temp_pi)

    plt.figure()
    plt.errorbar(perctimeopen, mean_percopenpref, yerr=sem_percopenpref, fmt='o')
    plt.xlabel('Percentage time spent in open arm,\nREAL data')
    plt.ylabel('Mean percentage of open-preferring cells,\n100 iterations, BOOTSTRAPPED data')

    # plt.figure()
    # plt.errorbar(perctimeopen, mean_percclosedpref, yerr=sem_percclosedpref, fmt='o')
    # plt.xlabel('Percentage time spent in open arm,\nREAL data')
    # plt.ylabel('Mean percentage of closed-preferring cells,\n100 iterations, BOOTSTRAPPED data')

    plt.figure()
    plt.errorbar(perctimeopen, mean_medpi, yerr=sem_medpi, fmt='o')
    plt.xlabel('Percentage time spent in open arm,\nREAL data')
    plt.ylabel('Mean of median preference index,\n100 iterations, BOOTSTRAPPED data')

    L = np.size(perctimeopen)

    plt.figure()
    plt.errorbar(perctimeopen[0:L:numepoch], mean_percopenpref[0:L:numepoch], yerr=sem_percopenpref[0:L:numepoch], fmt='o')
    plt.errorbar(perctimeopen[1:L:numepoch], mean_percopenpref[1:L:numepoch], yerr=sem_percopenpref[1:L:numepoch], fmt='o')
    plt.errorbar(perctimeopen[2:L:numepoch], mean_percopenpref[2:L:numepoch], yerr=sem_percopenpref[1:L:numepoch], fmt='o')
    plt.legend(('epoch 1', 'epoch 2', 'epoch 3'))
    plt.xlabel('Percentage time in open arms,\nREAL data')
    plt.ylabel('Mean percentage of open-preferring cells,\n100 iterations, BOOTSTRAPPED data')

    plt.figure()
    plt.errorbar(perctimeopen[0:L:numepoch], mean_medpi[0:L:numepoch], yerr=sem_medpi[0:L:numepoch], fmt='o')
    plt.errorbar(perctimeopen[1:L:numepoch], mean_medpi[1:L:numepoch], yerr=sem_medpi[1:L:numepoch], fmt='o')
    plt.errorbar(perctimeopen[2:L:numepoch], mean_medpi[2:L:numepoch], yerr=sem_medpi[2:L:numepoch], fmt='o')
    plt.legend(('epoch 1', 'epoch 2', 'epoch 3'))
    plt.xlabel('Percentage time in open arms,\nREAL data')
    plt.ylabel('Mean percentage of open-preferring cells,\n100 iterations, BOOTSTRAPPED data')

    plt.figure()
    for i in range(len(mousetuple)):
        plt.errorbar(perctimeopen[i * numepoch:(i + 1) * numepoch], mean_percopenpref[i * numepoch:(i + 1) * numepoch],
                     yerr=sem_percopenpref[i * numepoch:(i + 1) * numepoch], fmt='o')
    plt.legend(('m148', 'm149', 'm325', 'm2049', 'm2225', 'm2383'))
    plt.xlabel('Percentage time in open arms,\nREAL data')
    plt.ylabel('Mean percentage of open-preferring cells,\n100 iterations, BOOTSTRAPPED data')

    plt.figure()
    for i in range(len(mousetuple)):
        plt.errorbar(perctimeopen[i * numepoch:(i + 1) * numepoch], mean_medpi[i * numepoch:(i + 1) * numepoch],
                     yerr=sem_medpi[i * numepoch:(i + 1) * numepoch], fmt='o')
    plt.legend(('m148', 'm149', 'm325', 'm2049', 'm2225', 'm2383'))
    plt.xlabel('Percentage time in open arms,\nREAL data')
    plt.ylabel('Mean percentage of open-preferring cells,\n100 iterations, BOOTSTRAPPED data')


def cellsig_pca(rawcellsig, n_comp=10):

    from sklearn.decomposition import PCA

    zcellsig = sps.zscore(rawcellsig, axis=1)
    pca = PCA(n_components=n_comp)
    pcastats = pca.fit(zcellsig.T)
    pcacellsig = pca.transform(zcellsig.T).T

    return pcastats, pcacellsig

def probe_SVM_with_different_C_and_gamma_values_nonlinear_kernel(
        picklefile=r'I:\Inscopix_Data\all_mdict_nofilter.pickle', Cvals=(), gammavals=(), windowsize=40,
        fold=6, oversampleopt=False, niter=10):

    with open(picklefile, 'rb') as h:
        mdict = pickle.load(h)

    if not Cvals:
        Cvals = [10**i for i in range(-3,4)]
    if not gammavals:
        centergamma = 1/windowsize
        gammavals = [centergamma*10**i for i in range(-3,4)]

    columns = ['C', 'gamma'] + list(mdict.keys())
    index = range(len(Cvals)*len(gammavals))

    f1_df = pd.DataFrame(index=index, columns=columns)
    auc_df = pd.DataFrame(index=index, columns=columns)

    # midx = np.squeeze(np.where([bool(re.search('^m\d{2,4}', i)) for i in f1_df.columns]))

    idx = 0

    for i in mdict.keys():
        mdict[i].centerexit_df = mdict[i].get_center_exit_categories('whole', windowsize)

    # errors = np.empty((len(Cvals) * len(gammavals), len(mdict.keys())), dtype=object)

    for c in Cvals:

        for g in gammavals:

            print('Calculating values for C = {} and gamma = {}'.format(c, g))

            f1_df.loc[idx, 'C'] = c
            f1_df.loc[idx, 'gamma'] = g

            auc_df.loc[idx, 'C'] = c
            auc_df.loc[idx, 'gamma'] = g

            count = 0

            for i in mdict.keys():

                temp = mdict[i].run_svm_crossval(windowsize=windowsize, n_comp=5, C=c, gamma=g, kernel='rbf', fold=fold,
                                                 oversampleopt=oversampleopt, niter=niter)
                # if np.size(np.unique(temp[1])) == 2 and not np.isnan(temp[0]): #Only register results with both predictions
                f1_df.loc[idx, i] = np.mean(temp[0])
                auc_df.loc[idx, i] = np.mean(temp[1])

            # f1_df.loc[idx, 'mean'] = np.nanmean(f1_df.iloc[idx, midx])

                # if np.isnan(f1_df.loc[idx,i]):
                #     errors[idx, count] = np.nan
                # else:
                # errors[idx, count] = np.hstack((temp[1], temp[2]))

                count += 1

            idx += 1

    return f1_df, auc_df, mdict


def probe_SVM_with_different_C_values_linear_kernel(picklefile=r'I:\Inscopix_Data\all_mdict_nofilter.pickle', Cvals=(), windowsize=40, fold=5):

    with open(picklefile, 'rb') as h:
        mdict = pickle.load(h)

    if not Cvals:
        Cvals = [10**i for i in range(-3,4)]


    columns = ['C'] + list(mdict.keys())
    index = range(len(Cvals))

    f1_df = pd.DataFrame(index=index, columns=columns)

    # midx = np.squeeze(np.where([bool(re.search('^m\d{2,4}', i)) for i in f1_df.columns]))

    idx = 0

    for i in mdict.keys():
        mdict[i].centerexit_df = mdict[i].get_center_exit_categories('whole', windowsize)

    errors = np.empty((len(Cvals), len(mdict.keys()), 2))

    for c in Cvals:

        print('Calculating values for C = {}'.format(c))

        f1_df.loc[idx, 'C'] = c
        count = 0

        for i in mdict.keys():

            temp = mdict[i].run_svm_crossval(windowsize=windowsize, n_comp=5, C=c, kernel='linear', fold=fold)

            f1_df.loc[idx, i] = np.squeeze(temp[0])

            # temperrors = temp[2][np.squeeze(temp[1]) != temp[2]]
            # errors[idx, count, :] = [np.sum(temperrors==1)/np.sum(temp[2]==1), np.sum(temperrors==2)/np.sum(temp[2]==2)]
            count += 1

        idx += 1

    return f1_df, mdict


def plot_SVM_C_gamma_accuracy_matrix(SVM_df, plotopt=False):

    from palettable.colorbrewer.sequential import YlOrRd_9 as YOR9

    midx = np.squeeze(np.where([bool(re.search('^m\d{2,4}', i)) for i in SVM_df.columns]))
    Cunique, Cidx = np.unique(SVM_df.C, return_inverse=True)
    gammaunique, gammaidx = np.unique(SVM_df.gamma, return_inverse=True)

    Cgammadict = {}

    for i in midx:
        mousenum = SVM_df.columns[i]
        temp = np.asarray(SVM_df.iloc[:, i], dtype=float)

        Cgammamat = np.empty((np.size(Cunique), np.size(gammaunique)))

        for j,k,l in zip(Cidx, gammaidx, temp):
            Cgammamat[j,k] = l

        Cgammadict[mousenum] = Cgammamat

        if plotopt:
            fig = plt.figure()
            plt.imshow(Cgammadict[mousenum], cmap=YOR9.mpl_colormap, vmin=0, vmax=1, origin='lower')
            ax = fig.axes[0]
            a = ax.get_xticks().tolist()

            if len(a) != len(gammaunique):
                a[1:-1] = gammaunique
                ax.set_xticklabels(a)
            else:
                ax.set_xticklabels(gammaunique)

            a = ax.get_yticks().tolist()
            if len(a) != len(Cunique):
                a[1:-1] = Cunique
                ax.set_yticklabels(a)
            else:
                ax.set_yticklabels(Cunique)
            plt.xlabel('Gamma')
            plt.ylabel('C')
            plt.title(mousenum)
            plt.colorbar()
            plt.show()


    return Cgammadict, np.asarray(Cunique.reshape((np.size(Cunique), -1)), 'float'),\
           np.asarray(gammaunique.reshape((-1, np.size(gammaunique))), 'float')


def plot_SVM_score_vs_percopen(mdict, svm_score_dict, semilogopt=False, erroropt=True):

    logpercopen = np.empty((len(mdict.keys())))
    scoremean = np.empty((len(mdict.keys())))
    scoresem = np.empty((len(mdict.keys())))
    percopen = [i.percopen for i in mdict.values()]

    for c, i in enumerate(mdict.keys()):

        logpercopen[c] = np.log10(mdict[i].percopen)
        if isinstance(svm_score_dict[i], list):
            temp = [np.nanmean(j) for j in svm_score_dict[i]]
            scoremean[c] = np.nanmean(temp)
            scoresem[c] = sps.sem(temp, nan_policy='omit')
        elif isinstance(svm_score_dict[i], dict):
            scoremean[c] = np.nanmean(svm_score_dict[i])
            scoresem[c] = sps.sem(svm_score_dict[i], nan_policy='omit')
        # elif isinstance(svm_score_dict[i], pd.DataFrame): #for bestparams dataframe
        #     scoremean[c] = np.nanmean(svm_score_dict[i].loc[0,'score'])

    plt.figure()

    if semilogopt:
        if erroropt:
            plt.errorbar(logpercopen, scoremean, yerr=scoresem, fmt='o')
        else:
            plt.scatter(logpercopen, scoremean)
        regres = sps.linregress(logpercopen, scoremean)
        xi = np.linspace(np.min(logpercopen), np.max(logpercopen))
    else:
        if erroropt:
            plt.errorbar(percopen, scoremean, yerr=scoresem, fmt='o')
        else:
            plt.scatter(percopen, scoremean)
        regres = sps.linregress(percopen, scoremean)
        xi = np.linspace(np.min(percopen), np.max(percopen))

    yi = lambda y: regres.slope * y + regres.intercept
    plt.plot(xi, yi(xi), 'r--')
    xl = plt.xlim()
    yl = plt.ylim()
    plt.text(xl[0] + (xl[1] - xl[0]) * .75, yl[0] + (yl[1] - yl[0]) * .25,
             '$r^2$ = {:.3f}\np = {:.3f}'.format(regres.rvalue ** 2, regres.pvalue))
    if semilogopt:
        plt.xlabel('Log percentage time spent in open arm')
    else:
        plt.xlabel('Percentage time spent in open arm')
    plt.ylabel('SVM score')
    plt.show()

#
# def plot_SVM_score_vs_percopen(mdict, SVM_df):
#
#     boolidx = [bool(re.search('m\d{1,4}', i)) for i in SVM_df.columns]
#
#     maxpredacc = np.asarray(SVM_df.max(axis=0))[boolidx]
#     percopen = [i.percopen for i in mdict.values()]
#     logpercopen = np.log10(percopen)
#
#     plt.figure()
#     plt.scatter(logpercopen, maxpredacc)
#     regres = sps.linregress(logpercopen, maxpredacc)
#     xi = np.linspace(np.min(logpercopen), np.max(logpercopen))
#     yi = lambda y: regres.slope * y + regres.intercept
#     plt.plot(xi, yi(xi), 'r--')
#     xl = plt.xlim()
#     yl = plt.ylim()
#     plt.text(xl[0] + (xl[1] - xl[0]) * .75, yl[0] + (yl[1] - yl[0]) * .25,
#              '$r^2$ = {:.3f}\np = {:.3f}'.format(regres.rvalue ** 2, regres.pvalue))
#     plt.xlabel('Percentage time spent in open arm')
#     plt.ylabel('Percentage accuracy prediction by SVM')
#     plt.show()
#
#     # get percentage of open-arm runs
#     percopenruns = [np.sum(i.centerexit_df.run == 'open') / i.centerexit_df.shape[0] for i in mdict.values()]
#     plt.figure()
#     plt.scatter(percopenruns, maxpredacc)
#     plt.xlabel('Precentage of open-arm runs')
#     plt.ylabel('Percentage accuracy prediction by SVM')
#     plt.show()

    # plot errors
    # plt.figure()
    # idx = np.nanargmax(np.asarray(SVM_df.iloc[:, boolidx]), axis=0)
    # idx = (tuple(idx), tuple(range(idx.size)))
    # errorlist = [errors[x, y] for x, y in zip(idx[0], idx[1])]
    #
    # percclosederrors = []
    # percopenerrors = []
    #
    # for i in errorlist:
    #     errors = i[i[:, 0] != i[:, 1], 1]
    #     percclosederrors.append(np.sum(errors == 1)/np.sum(i[:,1] == 1))
    #     percopenerrors.append(np.sum(errors==2)/np.sum(i[:,1] == 2))
    #
    # for i, j in zip(percclosederrors, percopenerrors):
    #     plt.plot([i, j], color=(.8,.8,.8))
    #
    # plt.ylabel('Percentage error')
    # plt.xticks((0,1),('Closed run errors','Open run errors'))
    # plt.xlim((-.5, 1.5))
    # plt.show()
    #
    # # plot open arm errors
    # normopenerror = np.asarray(percopenerrors) * np.asarray(percopenruns)
    # plt.figure()
    # plt.scatter(percopen, normopenerror)
    # plt.xlabel('Percentage time spent in open arm')
    # plt.ylabel('Normalized percentage open-arm prediction accuracy by SVM')
    # plt.show()
    #
    # # plot closed arm errors
    # plt.figure()
    # plt.scatter(percopen, maxpredacc - normopenerror)
    # plt.xlabel('Percentage time spent in open arm')
    # plt.ylabel('Normalized percentage closed-arm prediction accuracy by SVM')
    # plt.show()


def get_best_SVM_parameters_for_each_mouse(SVM_df, num=5):

    boolidx = [bool(re.search('m\d{1,4}', i)) for i in SVM_df.columns]
    mice = SVM_df.columns[boolidx]

    bestparamdict = {}

    for m in mice:

        temp = SVM_df.loc[:,('C','gamma',m)].sort_values(by=m, ascending=False).reset_index(drop=True)
        bestparamdict[m] = temp.loc[:num-1, :].rename(columns={m:'score'})

    return bestparamdict

def pseudorandomly_split_training_CV_data(centerexit_df, fold=5, niter=10):

    oidx = np.squeeze(np.where(centerexit_df.loc[:,'run'] == 'open'))
    cidx = np.squeeze(np.where(centerexit_df.loc[:,'run'] == 'closed'))
    groups = []

    for i in range(niter):

        np.random.shuffle(oidx)
        ogroups = np.array_split(oidx, fold)
        np.random.shuffle(ogroups)

        np.random.shuffle(cidx)
        cgroups = np.array_split(cidx, fold)
        np.random.shuffle(cgroups)

        groups.append([np.hstack((i,j)) for i,j in zip(ogroups,cgroups)])

    return groups

def get_nonoverlapping_closed_open_segments(anymaze_ts, seglength=20, samples=50, ignorethresh=200):

    # get all closed/open samples
    allclosedsamples = np.asarray(
        anymaze_ts[(anymaze_ts.loc[:, 'status'] == 'closed') | (anymaze_ts.loc[:, 'status'] == 'approach')].index)
    allopensamples = np.setdiff1d(range(10800), allclosedsamples)

    # remove first 10 seconds
    allclosedsamples = allclosedsamples[allclosedsamples >= ignorethresh]
    allopensamples = allopensamples[allopensamples >= ignorethresh]

    # get random closed indices
    idx = np.diff(allclosedsamples) == 1
    rlencode = jsbox.rude(idx)
    cumidx = np.cumsum(rlencode[1, :])
    seqstartidx = cumidx[np.squeeze(np.where(rlencode[1, :] > seglength)) - 1]
    if seqstartidx[0] > seqstartidx[1]:  # for indices starting at 0
        seqstartidx[0] = 0
    seqendidx = cumidx[np.squeeze(np.where(rlencode[1, :] > seglength))]
    allidx = np.asarray(
        [index for k in [list(range(i, j - seglength + 2)) for i, j in zip(seqstartidx, seqendidx)] for index in k])

    closedsamples = []
    for i in range(samples):
        tempidx = np.random.choice(allidx)
        closedsamples.append(allclosedsamples[tempidx])

        delidx = np.in1d(allclosedsamples, range(allclosedsamples[tempidx] - seglength + 1,
                                                 allclosedsamples[tempidx] + seglength)).nonzero()
        delidxidx = np.in1d(allidx, delidx).nonzero()

        try:
            allidx = np.delete(allidx, delidxidx)
        except ValueError:
            print('Only managed to get {} closed samples!'.format(i))
            break

    # get random open indices
    idx = np.diff(allopensamples) == 1
    rlencode = jsbox.rude(idx)
    cumidx = np.cumsum(rlencode[1, :])
    seqstartidx = cumidx[np.squeeze(np.where(rlencode[1, :] > seglength)) - 1]
    if seqstartidx[0] > seqstartidx[1]:  # for indices starting at 0
        seqstartidx[0] = 0
    seqendidx = cumidx[np.squeeze(np.where(rlencode[1, :] > seglength))]
    allidx = np.asarray(
        [index for k in [list(range(i, j - seglength + 2)) for i, j in zip(seqstartidx, seqendidx)] for index in k])

    opensamples = []
    for i in range(samples):
        tempidx = np.random.choice(allidx)
        opensamples.append(allopensamples[tempidx])

        delidx = np.in1d(allopensamples, range(allopensamples[tempidx] - seglength + 1,
                                               allopensamples[tempidx] + seglength)).nonzero()
        delidxidx = np.in1d(allidx, delidx).nonzero()

        try:
            allidx = np.delete(allidx, delidxidx)
        except ValueError:
            print('Only managed to get {} open samples!'.format(i))
            break

    return closedsamples, opensamples

def batch_save_mdict(outfile, folder=r'I:\Inscopix_Data', threshtype='prctile', thresh=95, ignoreopt=True,
                     filteropt=False, cellsigopt='raw', zscoreopt=True):

    files = glob.glob(os.path.join(folder,'m*.pickle'))
    mdict = {}

    for f in files:
        mnum = re.search('m\d{3,4}(?=_RTEPM)', f).group(0)

        if threshtype == 'prctile':
            mdict[mnum] = inscopix(f, prctilethresh=thresh, ignoreopt=ignoreopt, filteropt=filteropt, cellsigopt=cellsigopt, zscoreopt=zscoreopt)
        elif threshtype == 'std':
            mdict[mnum] = inscopix(f, stdthresh=thresh, ignoreopt=ignoreopt, filteropt=filteropt, cellsigopt=cellsigopt, zscoreopt=zscoreopt)
        else:
            raise Exception('threshtype can only be ''std'' or ''prctile''!')

    with open(outfile,'wb') as of:
        pickle.dump(mdict, of)


def circularly_shuffle_cellsig(cellsig, shiftidx=()):

    numcells, nbins = cellsig.shape
    shuffcellsig = np.empty(cellsig.shape)

    if not shiftidx:
        shiftidx = np.random.randint(1, nbins, numcells)

    elif np.isscalar(shiftidx):
        shiftidx = np.repeat(int(shiftidx), numcells)

    elif shiftidx.ndim == 1 and shiftidx.size == cellsig.shape[0]:
        pass

    else:
        raise Exception('Something''s wrong!')

    for i in range(numcells):
        shuffcellsig[i,:] = np.roll(cellsig[i,:], shiftidx[i])

    return shuffcellsig


def get_run_sequences(ts, tstype='whole', tslength=40):

    assert (tstype == 'snippets' or tstype == 'whole')

    tsstatus = ts.status
    rlestatus = jsbox.rude(tsstatus)

    # combine approach/center
    for i in range(np.size(rlestatus,1)):
        if rlestatus[0,i] == 'approach' or rlestatus[0,i] == 'center':
            rlestatus[0,i] = 'decision'

    # remove very short close and open time series (1s) and reassign to center to remove transition noise
    for i in range(np.size(rlestatus,1)):
        if (rlestatus[0,i] == 'closed' or rlestatus[0,i] == 'open') and np.int(rlestatus[1,i]) < 20:
            rlestatus[0,i] = 'decision'
    temp = jsbox.rude(rlestatus)
    rlestatus = jsbox.rude(temp)


    # get cumulative idx and labels
    count = np.asarray(rlestatus[1,:], 'int')
    idx = np.cumsum(count)
    labels = rlestatus[0,:]

    # get indices of closed-center-closed and closed-center-open sequences (+1 to get index of center entry)
    # cc_run = jsbox.search_sequence_numpy(labels, np.array(['closed','decision','closed'])) + 1
    # co_run = jsbox.search_sequence_numpy(labels, np.array(['closed','decision','open'])) + 1

    cc_run = jsbox.search_sequence_numpy(labels, np.array(['decision','closed']))
    co_run = jsbox.search_sequence_numpy(labels, np.array(['decision','open']))

    if tstype == 'snippets':

        # Get center time
        cc_run_len = np.asarray(rlestatus[1, cc_run], 'int')
        co_run_len = np.asarray(rlestatus[1, co_run], 'int')

        # Get number of snippets for each run and mod for deciding to round up or down
        cc_divmod = np.divmod(cc_run_len, tslength)
        co_divmod = np.divmod(co_run_len, tslength)

        # Number of snippets for each run
        cc_seg = np.asarray([i + 1 if j > 10 else i for i, j in zip(cc_divmod[0], cc_divmod[1])])
        co_seg = np.asarray([i + 1 if j > 10 else i for i, j in zip(co_divmod[0], co_divmod[1])])

    elif tstype == 'whole':
        cc_seg = np.asarray(np.ones((1, cc_run.size)), int)
        co_seg = np.asarray(np.ones((1, co_run.size)), int)

    # Initialize dataframe
    centerexit_df = pd.DataFrame(index=list(range(np.sum(cc_seg)+ np.sum(co_seg))),
                                 columns=('idx','run','stim'))

    # Set start and end indices for closed runs
    cc_run_end_idx = idx[cc_run]
    cc_run_start_idx = cc_run_end_idx - cc_seg * tslength

    # Split closed runs up into snippets (if necessary)
    if tstype == 'snippets':
        cc_start_idx = np.hstack([np.arange(i, j, tslength) for i, j in zip(cc_run_start_idx, cc_run_end_idx)])
    else:
        cc_start_idx = cc_run_start_idx

    # Save data in df
    centerexit_df.loc[range(np.size(cc_start_idx)), 'idx'] = cc_start_idx
    centerexit_df.loc[range(np.size(cc_start_idx)), 'run'] = 'closed'

    # Same as above but for open runs
    co_run_end_idx = idx[co_run]
    co_run_start_idx = co_run_end_idx - co_seg * tslength

    if tstype == 'snippets':
        co_start_idx = np.hstack([np.arange(i, j, tslength) for i, j in zip(co_run_start_idx, co_run_end_idx)])
    else:
        co_start_idx = co_run_start_idx

    centerexit_df.loc[range(np.size(cc_start_idx), np.size(centerexit_df,0)), 'idx'] = co_start_idx
    centerexit_df.loc[range(np.size(cc_start_idx), np.size(centerexit_df,0)), 'run'] = 'open'

    # Determine which are stim on vs stim off runs
    stimonstart = 3600-10
    stimonend = 7200-10

    # Populate stim column based on index
    centerexit_df['stim'].mask(centerexit_df['idx'].apply(lambda x: stimonstart <= x <= stimonend),
                               'on', inplace=True)
    centerexit_df['stim'].fillna('off',inplace=True)

    return centerexit_df

def run_inscopix_svm(centerexit_df, rawcellsig, windowsize=40, n_comp=5, C=1, gamma='auto', kernel='rbf', fold=1,
                     niter=10, oversampleopt=False):

    """General/base SVM code for inscopix data"""

    from sklearn import svm
    from sklearn import metrics
    from imblearn.over_sampling import SMOTE

    assert fold >= 1

    centerexit_df = centerexit_df.drop(centerexit_df.index[centerexit_df['idx'] < 1])

    # sort and reindex (shouldn't really matter)
    # centerexit_df.sort_values(by='idx', inplace=True)
    centerexit_df.reset_index(drop=True, inplace=True)

    pcastats, pcacellsig = cellsig_pca(rawcellsig, n_comp=n_comp)

    trainclass = list(centerexit_df.run)
    all_y = np.asarray([1 if x == 'open' else 0 for x in trainclass])

    if fold == 1:
        groups = [(np.arange(centerexit_df.shape[0]))]
        niter = 1
    else:
        groups = pseudorandomly_split_training_CV_data(centerexit_df, fold, niter)

    testy = np.empty((centerexit_df.shape[0], niter))
    f1score = np.empty((niter, 1))
    auc = np.empty((niter, 1))
    scorey = np.empty((centerexit_df.shape[0], niter))

    for n in range(niter):

        for test_i in groups[n]:

            train_i = np.setdiff1d(range(centerexit_df.shape[0]), test_i)
            trainstartidx = np.asarray(centerexit_df.loc[train_i, 'idx'] - 1)
            trainendidx = trainstartidx + windowsize + 1

            y = all_y[train_i]

            X = np.empty((train_i.size, int(windowsize * n_comp)))
            ratio = np.mean(y == 0)

            for index in range(trainstartidx.size):
                X[index, :] = np.asarray([np.diff(pcacellsig[x, trainstartidx[index]:trainendidx[index]])
                                          for x in range(n_comp)]).flatten('C')

            if kernel == 'rbf':
                if oversampleopt:
                    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
                else:
                    clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, class_weight={0: 1 - ratio, 1: ratio})
            elif kernel == 'linear':
                if oversampleopt:
                    clf = svm.LinearSVC(C=C)
                else:
                    clf = svm.LinearSVC(C=C, class_weight={0: 1 - ratio, 1: ratio})
            else:
                raise Exception('Only rbf and linear kernels supported right now!')
            # clf = RandomForestClassifier(class_weight={1:1-ratio, 2:ratio})

            if oversampleopt:

                minkneigh = np.min((np.sum(y == 1), np.sum(y == 0)))

                if minkneigh <= 5:
                    sm = SMOTE(k_neighbors=minkneigh - 1)

                else:
                    sm = SMOTE()

                sm = sm.fit(X, y)
                X_resample, y_resample = sm.sample(X, y)
                clf.fit(X_resample, y_resample)

            else:
                clf.fit(X, y)

            teststartidx = np.asarray(centerexit_df.loc[test_i, 'idx'] - 1)
            testendidx = teststartidx + windowsize + 1

            if fold == 1:
                testX = np.asarray([np.diff(pcacellsig[x, teststartidx:testendidx]) for x in range(n_comp)]).flatten(
                    'C').reshape(1, -1)

            else:
                testX = np.empty((test_i.size, int(windowsize * n_comp)))

                for index in range(teststartidx.size):
                    testX[index, :] = np.asarray([np.diff(pcacellsig[x, teststartidx[index]:testendidx[index]])
                                                  for x in range(n_comp)]).flatten('C')

            testy[test_i, n] = clf.predict(testX)
            scorey[test_i, n] = clf.decision_function(testX)

        if np.size(np.unique(testy[:, n])) == 1:
            f1score[n, 0] = np.nan
            auc[n, 0] = np.nan
        else:
            f1score[n, 0] = metrics.f1_score(all_y, testy[:, n])
            auc[n, 0] = metrics.roc_auc_score(all_y, scorey[:, n])

    return f1score, auc, testy, all_y, scorey


def plot_closed_open_SVM_accuracy(pred_y, actual_y):

    plt.figure()
    openacc = {}
    closedacc = {}

    for i in pred_y.keys():

        actual = actual_y[i]
        predicted = pred_y[i]

        closedidx = actual == 0
        openidx = actual == 1

        tempclosed = np.empty((predicted.shape[1]))
        tempopen = np.empty((predicted.shape[1]))

        for j in range(predicted.shape[1]):
            trialpred = predicted[:, j]
            tempclosed[j] = np.sum(trialpred[closedidx] == 1) / trialpred[closedidx].size
            tempopen[j] = np.sum(trialpred[openidx] == 0) / trialpred[openidx].size

        closedacc[i] = tempclosed
        openacc[i] = tempopen

        acc = np.vstack((tempclosed, tempopen))
        plt.errorbar(range(2), np.mean(acc, axis=1), yerr=sps.sem(acc, axis=1), Color='xkcd:grey',
                     ecolor='xkcd:grey', zorder=3)

    plt.xticks((0, 1), ('Closed run errors', 'Open run errors'))
    plt.xlim((-.5, 1.5))

    openmn = [np.mean(i) for i in openacc.values()]
    closedmn = [np.mean(i) for i in closedacc.values()]
    __, pval = sps.ttest_rel(closedmn, openmn)
    plt.figtext(0.45, 0.2, 'p = {:.2f}'.format(pval))
    plt.ylabel('Percentage error')

    plt.show()

    return closedacc, openacc

def plot_open_pred_accuracy_over_runs(pred_y, actual_y):

    plt.figure()
    xaxis = (0, 1)

    for i in pred_y.keys():

        actual = actual_y[i]
        openidx = actual == 1

        openpred = pred_y[i][openidx]
        meanacc = np.mean(openpred, axis=1)
        halfidx = int(np.ceil(meanacc.size/2))

        meanacc = (np.mean(meanacc[0:halfidx]), np.mean(meanacc[halfidx:]))

        plt.plot(xaxis, meanacc)

    plt.show()

def calc_inscopix_trace_running_mean(cellsig, window=200):

    runningmean = np.empty(cellsig.shape)

    for i in range(cellsig.shape[1]):
        startidx = i - window
        endidx = i + window + 1

        if startidx < 0:
            startidx = 0
        if endidx > cellsig.shape[1]:
            endidx = cellsig.shape[1]

        runningmean[:,i] = np.mean(cellsig[:, startidx:endidx], axis=1)

    return runningmean


def calc_epoch_pref_idx(cellsig, anymaze, posdf, stdthresh=(), prctilethresh=(), filteropt=False, zscoreopt=True,
                        cellsigopt='raw'):

    if not stdthresh and not prctilethresh:
        raise Exception('Enter a value for stdthresh or prctilethresh')
    elif np.isscalar(stdthresh) and np.isscalar(prctilethresh):
        raise Exception('Enter a value for either stdthresh or prctilethresh only')

    warnings.simplefilter('ignore', RuntimeWarning)
    boundaryidx = np.linspace(0, np.size(cellsig, 1), 4, dtype=int)

    # Process some basic information from create_plusmazemap
    pos = np.array(anymaze.loc[:,['CentrePosnY','CentrePosnX']]).T
    maxypos = np.max(pos[0,:])
    maxxpos = np.max(pos[1,:])

    startthresh = 10  # ignore the first 10 seconds of experiment
    startidx = int(np.squeeze(np.where(anymaze['Time'] == startthresh)))

    print('Creating plusmazemap for epochs...')

    calciummap = []
    calciummap_thresh = []
    prefdf = []

    for i in range(boundaryidx.size-1):

        print('\nProcessing epoch {}'.format(i+1))

        if i != 0:
            startidx = boundaryidx[i]

        endidx = boundaryidx[i+1]


        # Normalize cell sig if necessary
        if zscoreopt:
            zcellsig = sps.zscore(cellsig[:,startidx:endidx], 1)
        else:
            zcellsig = cellsig[:,startidx:endidx]

        if cellsigopt == 'derivative':
            proc_cellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
            proc_cellsig[proc_cellsig < 0] = 0
        elif cellsigopt == 'derivative_product':
            dcellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
            dcellsig[dcellsig <= 0] = 0
            proc_cellsig = zcellsig * dcellsig
        elif cellsigopt == 'raw':
            proc_cellsig = zcellsig
        elif cellsigopt == 'raw_product':
            dcellsig = np.hstack(((np.zeros((zcellsig.shape[0], 1))), np.diff(zcellsig)))
            dcellsig[dcellsig <= 0] = 0
            dcellsig[dcellsig > 0] = 1
            proc_cellsig = zcellsig * dcellsig
        elif cellsigopt == 'df':
            blcellsig = calc_inscopix_trace_running_mean(zcellsig, window=200)
            proc_cellsig = zcellsig - blcellsig
        elif cellsigopt == 'df/f':
            blcellsig = calc_inscopix_trace_running_mean(zcellsig, window=200)
            assert (all(np.min(blcellsig, axis=1) > 0))
            proc_cellsig = (zcellsig - blcellsig) / blcellsig
        else:
            raise Exception('Invalid cellsigopt!')

        temppos = pos[:,startidx:endidx]

        heatmap = np.zeros((maxypos + 10, maxxpos + 10))

        for j in range(temppos.shape[1]):
            heatmap[temppos[0, j], temppos[1, j]] += 1

        calciummaps = calculate_calciummaps(proc_cellsig, heatmap, temppos, prctilethresh=prctilethresh,
                                            filteropt=filteropt, maxxpos=maxxpos, maxypos=maxypos)

        calciummap.append(calciummaps[0])
        calciummap_thresh.append(calciummaps[1])

        prefdf.append(classify_cells_by_sig_pixels(posdf, calciummap_thresh[i]))

    return calciummap, calciummap_thresh, prefdf


def plot_perc_open_pref_idx_scatter_plot(prefidxdict, percopendict, plottype='mean'):

    prefidx = []
    percopen = []

    for k in prefidxdict.keys():

        for idx in range(len(prefidxdict[k])):

            if plottype == 'all':
                temppi = np.array(prefidxdict[k][idx].loc[:, 'prefidx'])
                prefidx.append(temppi)
                percopen.append(np.tile(percopendict[k][idx], (1, temppi.size)))
            elif plottype == 'mean':
                prefidx.append(np.mean(prefidxdict[k][idx].loc[:, 'prefidx']))
                percopen.append(percopendict[k][idx])

    if plottype == 'all':
        prefidx = np.concatenate(prefidx)
        percopen = np.concatenate(percopen, axis=1).ravel()

    plt.figure()
    plt.scatter(percopen, prefidx)
    plt.xlabel('Percentage time spent in open arms')
    plt.ylabel('Preference index')


def plot_SVM_score_vs_perc_cells(picklefile, plotopt=True):

    with open(picklefile, 'rb') as f:
        SVMscore = pickle.load(f)[1]

    perc = [np.around(i, 1) for i in list(SVMscore.keys())]
    score = np.empty((np.size(perc), 1))
    sem = np.empty((np.size(perc), 1))

    for i,k in enumerate(SVMscore.keys()):

        scorearray = [np.nanmean(ss) for ss in SVMscore[k]]

        score[i] = np.nanmean(scorearray)
        sem[i] = np.nanstd(scorearray) / np.size(scorearray)

    mouse = re.search('(?<=\\\)\w+(?=.pickle)', picklefile).group(0)

    if plotopt:

        # plt.figure()
        # plt.plot(perc, score, 'bx-')
        plt.errorbar(perc, score, yerr=sem, Color='k', ecolor='k', zorder=3)
        plt.xlabel('Ratio of neurons')
        plt.ylabel('SVM score')
        plt.title('{} SVM score vs ratio of neurons'.format(mouse))

    return perc, score, sem

def batch_plot_SVM_score_vs_perc_cells(picklefolder, plotallopt=False):

    picklefiles = glob.glob(os.path.join(picklefolder, '*.pickle'))
    allscores = []

    for pf in picklefiles:
        allscores.append(plot_SVM_score_vs_perc_cells(pf, plotallopt))

    normscores = np.empty((len(picklefiles), np.shape(allscores[0][0])[0]))

    plt.figure()

    for i,score in enumerate(allscores):
        normscores[i,:] = (score[1] / score[1][-1]).T
        plt.plot(score[0], normscores[i,:], color=(.8,.8,.8))

    meanscores = np.mean(normscores, axis=0)
    semscores = np.std(normscores, axis=0) / np.shape(normscores)[0]
    plt.errorbar(allscores[0][0], meanscores, yerr=semscores, Color='k', ecolor='k', zorder=3)


def plot_prediction_acc_vs_pref_idx_spread(SVMfile, PIfile):

    # from mpl_toolkits.mplot3d import Axes3D

    with open(SVMfile, 'rb') as f:
        SVMscore = pickle.load(f)

    PI = inscopix(PIfile, prctilethresh=95)

    prefidx = np.array(PI.prefdf.loc[:, 'prefidx'])

    c = 1
    plt.figure()

    for svm in SVMscore[0].keys():

        niter = len(SVMscore[0][svm])
        allscores = np.empty((niter, 1))
        allmean = np.empty((niter, 1))
        allstd = np.empty((niter, 1))

        for i, (neurons, score) in enumerate(zip(SVMscore[0][svm], SVMscore[1][svm])):
            prefindices = [prefidx[n] for n in neurons]
            allscores[i] = np.nanmean(score)
            allmean[i] = np.mean((prefindices))
            # allstd[i] = np.var(prefindices) / np.mean(prefindices)

        plt.subplot(2,2,c)
        plt.scatter(allmean, allscores)
        plt.xlabel('Preference index mean')
        plt.ylabel('SVM score')
        # ax.set_ylabel('Fano factor')
        # ax.set_zlabel('SVM score')
        plt.title('{} percent of cells'.format(int(svm*100)))

        if c%4 == 0:
            plt.figure()
            c = 1
        else:
            c += 1

def plot_SVM_score_by_pref_idx_groups(files):

    with open(files[0], 'rb') as fidin:
        temp = pickle.load(fidin)

    numgroups = len(temp[0].keys())
    medians = np.empty((len(files), numgroups))

    plt.figure()
    c = 1

    for n,f in enumerate(files):

        mouse = re.search('m\d{3,4}(?=.pickle)', f).group(0)

        with open(f, 'rb') as fidin:
            datadict = pickle.load(fidin)

        groupscore = pd.DataFrame(columns=('SVMscore', 'group'))
        numunits = np.size(datadict[0][next(iter(datadict[0]))][0])

        for i,k in enumerate(datadict[1].keys()):

            temp = pd.DataFrame(columns=('SVMscore', 'group'))

            # for t in datadict[1][k]:
            #     if np.isnan(t).all():
            #         raise Exception('Stop!')


            temp.loc[:, 'SVMscore'] = [np.nanmean(i) for i in datadict[1][k]]
            temp.loc[:, 'group'] = k
            medians[n,i] = np.nanmedian(temp['SVMscore'])
            groupscore = groupscore.append(temp, ignore_index=True)

        plt.subplot(2,2,c)
        sns.swarmplot(x='group', y='SVMscore', data=groupscore)
        sns.boxplot(x='group', y='SVMscore', data=groupscore)
        plt.title('{}, n = {}'.format(mouse, numunits))

        if c % 4 == 0 & n != len(files):
            plt.figure()
            c = 1
        else:
            c += 1


def get_mouse_specific_SVM_score_tables(SVM_df):

    mousenum = list(SVM_df.columns[[bool(re.search('m\d{1,4}', i)) for i in SVM_df.columns]])

    Cvals = np.array(np.unique(SVM_df['C']), dtype=float)
    gammavals = np.array(np.unique(SVM_df['gamma']), dtype=float)
    defaultdf = pd.DataFrame(index=Cvals, columns=gammavals)

    SVM_mdict = {}

    for mn in mousenum:
        temp = copy.deepcopy(defaultdf)

        for c in Cvals:
            for g in gammavals:
                temp.loc[c, g] = float(SVM_df.loc[(SVM_df['C'] == c) & (SVM_df['gamma'] == g), mn])

        SVM_mdict[mn] = temp

    return SVM_mdict


    # plt.figure()
    # for m in medians:
    #     plt.plot(range(3), m, color=(.8,.8,.8), marker='.')
    #
    # totalmn = np.mean(medians, axis=0)
    # totalsem = np.std(medians, axis=0) / np.shape(medians)[0]
    # plt.errorbar(range(3), totalmn, yerr=totalsem, Color='k', ecolor='k', zorder=3)

# def plot_shuffprefidx_histogram(permprefidx, prefidx):
#
#     permpref = np.empty((len(permprefidx), permprefidx[0].shape[0]))
#
#     for idx in range(len(permprefidx)):
#         permpref[idx,:] = permprefidx[idx].loc[:, 'prefidx']



## Check for huge negative inflections in calcium signal
## Shuffle Nick Frost style.