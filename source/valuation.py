__author__ = 'Herakles II'

# Imports
import numpy as np


INF = 1e20

def interpolate(x, y, x_new, offset=np.zeros(1), penalty=[-INF, -INF], mode='cross'):
    # mode can be cross or straight    
    n_l = len(x_new)
    n_a = len(offset)
    
    if mode is 'cross':
        y_int = np.zeros((n_l, y.shape[0], n_a))
    else:
        y_int = np.zeros((n_l, n_a))

    x_min = x.min()
    x_max = x.max()
    for a in range(n_a):
        x1 = x_new + offset[a]
        ind_below = x1 < x_min
        ind_above = x1 > x_max
        ind_isin = np.bitwise_not(np.logical_or(ind_below, ind_above))
        position = x1[:, None] > x[None, :]
        level = np.argmin(position, axis=1)
        ind_first = np.logical_and(level == 0, ind_isin)
        ind_between = np.logical_and(np.bitwise_not(ind_first), ind_isin)
        maps = level[ind_between]
        lbd = (x1[ind_between] - x[maps-1])/(x[maps]-x[maps-1])
        if mode is 'cross':
            y_int[ind_below, :, a] = penalty[0]
            y_int[ind_above, :, a] = penalty[1]
            y_int[ind_first, :, a] = y[:, 0].T
            y_int[ind_between, :, a] = (y[:, maps-1]*(1-lbd) + y[:, maps]*lbd).T
        else:
            y_int[ind_below, a] = penalty[0]
            y_int[ind_above, a] = penalty[1]
            y_int[ind_first, a] = y[ind_first, 0]
            y_int[ind_between, a] = (y[ind_between, maps-1]*(1-lbd) + y[ind_between, maps]*lbd).T

    return y_int

def designMatrix(x, threshold=(1/np.finfo(np.float64).eps)/1000, max_degree=1):
    num_sims, num_factors = x.shape
    X = np.ones(shape=(num_sims, 1))
    XTX = np.array([[num_sims]])
    for p in range(1, max_degree+1):
        for k in range(p+1):
            for i in range(num_factors):
                for j in range(num_factors):
                    factor = (x[:, i]**k * x[:, j]**(p-k))[:, None]
                    XTx = np.dot(X.T, factor)
                    xTx = np.dot(factor.T, factor)
                    XTXnew = np.concatenate((np.concatenate((XTX, XTx), axis=1),
                                             np.concatenate((XTx.T, xTx), axis=1)),
                                            axis=0)
                    if np.linalg.cond(XTXnew) < threshold:
                        X = np.concatenate((X, factor), axis=1)
                        XTX = XTXnew
    return X, XTX

def get_expected_value(x, Y):
    X, XTX = designMatrix(x)
    invXTX = np.linalg.inv(XTX)
    solMat = np.dot(invXTX, X.T)
    Coeff = np.dot(solMat, Y)
    Val = np.dot(X, Coeff.reshape((X.shape[1], Y.shape[0]*Y.shape[2])))
    return Val.reshape((Y.shape[1], Y.shape[0],Y.shape[2]))

class swing(object):
    def __init__(self, **kwargs):
        # ToDo: Init all variables here!
        self.num = dict()
        self.params = dict()        
        self.volume_grid = dict()
        self.value = dict()
        self.delta = None
        self.cum_cq = dict()
        
        self.set_attributes(**kwargs)
        if 'model' in kwargs.keys():
            self.set_model(kwargs['model'])

        self.set_parameters(**kwargs)
        self.test_quantities()
        self.obtain_cumulative_contract_quantities()
        self.obtain_volume_grid()
        
    def set_model(self, model):
        if model is 'oqt':
            self.set_process = self._oqt_set_process
            self.obtain_stopping_scheme = self._oqt_obtain_stopping_scheme
            self.obtain_delta = self._oqt_obtain_delta
        elif model is 'lsmc':
            self.set_process = self._lsmc_set_process
            self.obtain_stopping_scheme = self._lsmc_obtain_stopping_scheme
            self.obtain_delta = self._lsmc_obtain_delta
        else:
            raise Exception('No such model with name ' +str(model))
        self.model = model
        
        
    def set_attributes(self, cq_max=None, cq_min=None, dq_max=None,
                       num_steps=None, dq_min=0.0, **kwargs):
        # INPUT
        # =====
        # cq_max        - natural number
        # cq_min        - natural number
        #               > must be smaller than cq_max
        # dq_max        - real number or array of size (num['steps'])
        # dq_min (Opt.) - real number or array of size (num['steps'])
        #               > Default value is zero

        self.num['steps'] = num_steps
        self.cq = {'max': np.float(cq_max), 
                   'min': np.float(cq_min)}
        self.dq = {'max': np.float(dq_max)*np.ones(self.num['steps']), 
                   'min': np.float(dq_min)*np.ones(self.num['steps'])}
        
    def set_parameters(self, relative_spacing=2.0, num_actions=2, **kwargs):
        # INPUT
        # =====
        # relative_spacing    - real number
        # num_actions         - natural number
        #                     > number of possible actions per step
        
        self.params['spacing'] = relative_spacing
        self.num['actions'] = num_actions

    def test_quantities(self):
        if not self.dq['max'].sum() >= self.cq['min']:
            raise Exception('Maximal daily quantities are too low. '
                            'Can not reach minimal contract quantity!')
        if not self.dq['min'].sum() <= self.cq['max']:
            raise Exception('Minimal daily quantities are too high. '
                            'Can not get below maximal contract quantity!')

    def obtain_cumulative_contract_quantities(self):
        forward = np.zeros(self.num['steps']+1)
        forward[1:(self.num['steps']+1)] += self.dq['max'].cumsum()
        backward = self.cq['max']*np.ones(self.num['steps']+1)
        backward[0:self.num['steps']] -= self.dq['min'][::-1].cumsum()[::-1]

        self.cum_cq['max'] = np.minimum(forward, backward)

        forward = np.zeros(self.num['steps']+1)
        forward[1:(self.num['steps']+1)] += self.dq['min'].cumsum()
        backward = self.cq['min']*np.ones(self.num['steps']+1)
        backward[0:self.num['steps']] -= self.dq['max'][::-1].cumsum()[::-1]

        self.cum_cq['min'] = np.maximum(forward, backward)

    def obtain_volume_grid(self):        
        levels = (self.num['steps']+1)*['']
        dq_mean = (self.dq['max']-self.dq['min']).mean() # ToDo: This could be more sophisticated!
        relative_change = (self.cum_cq['max']-self.cum_cq['min'])/dq_mean/self.params['spacing']
        self.volume_grid['size'] = (np.ceil(relative_change)+1).astype(np.int64)
        
        for step in range(self.num['steps']+1):
            levels[step] = np.linspace(self.cum_cq['min'][step],
                                          self.cum_cq['max'][step],
                                          num=self.volume_grid['size'][step])
        self.volume_grid['levels'] = levels
        
        delta = np.linspace(0, 1, num=self.num['actions'])
        
        changes_pos = (self.num['steps'])*['']
        for step in range(self.num['steps']):
            num_levels = len(self.volume_grid['levels'][step])
            changes = np.reshape(np.repeat(self.dq['max'][step]*delta 
                                                 + self.dq['min'][step]*(1-delta), num_levels), 
                                    (self.num['actions'], num_levels))
            new_levels = self.volume_grid['levels'][step] + changes
            under = new_levels < self.cum_cq['min'][step+1]
            above = new_levels > self.cum_cq['max'][step+1]
            
            changes[under] -= (new_levels-self.cum_cq['min'][step+1])[under]
            changes[above] -= (new_levels-self.cum_cq['max'][step+1])[above]
            changes_pos[step] = changes
        self.volume_grid['changes'] = changes_pos
        
    def _oqt_set_process(self, oqt):
        # INPUT
        # =====
        # oqt    - dict with keys [centers, mapping, transitions]
        
        if (not oqt.centers.shape[0] == self.num['steps']
            or not oqt.probabilities.shape[0] == self.num['steps']
            or not oqt.transitions.shape[0]+1 == self.num['steps']):
            raise Exception('Given tree does not fit to the set number of steps')
        if (not oqt.centers.shape[1] == oqt.probabilities.shape[1]
            or not oqt.centers.shape[1] == oqt.transitions.shape[1]
            or not oqt.centers.shape[1] == oqt.transitions.shape[2]):
            raise Exception('Given tree does not have consistent number of clusters')
        
        if not len(oqt.centers.shape) > 2:
            oqt.centers = oqt.centers[:,:,None]
        self.num = oqt.num
        self.oqt = oqt
        
    def _oqt_obtain_stopping_scheme(self):
        centers = self.oqt.centers
        transitions = self.oqt.transitions
        
        grid_size_last = self.volume_grid['size'][-1]
        self.stopping_scheme = self.num['steps']*['']
        values = np.zeros((self.num['clusters'], grid_size_last))
        
        for step in range(self.num['steps']-1, -1, -1):
            values_int = interpolate(x=self.volume_grid['levels'][step+1], 
                                     y=values, 
                                     x_new=self.volume_grid['levels'][step],
                                     offset=self.volume_grid['changes'][step])
            if step < self.num['steps']-1:
                values_new = np.dot(transitions[step, :, :], values_int)
            else:
                values_new = np.rollaxis(values_int, axis=1)
            payout = centers[step, :, 0, None, None]*self.volume_grid['changes'][step][ :, :, None].T
            values_with_payout = values_new + payout
            best_strategie = np.argmax(values_with_payout, axis=2)
            self.stopping_scheme[step] = self.volume_grid['changes'][step][best_strategie, range(self.volume_grid['size'][step])]
                                                       
            
            values = np.max(values_with_payout, axis=2) #ToDo: This could be faster, since the maximal element is already obtained two lines above.
        self.value['backward'] = np.dot(self.oqt.probabilities[0], values[:, 0])
        
    def _oqt_obtain_delta(self):        
        transitions = self.oqt.transitions
        probabilities = self.oqt.probabilities
        
        stopping = np.zeros((self.num['steps'], self.num['clusters']))
        volume_cum = np.zeros(self.num['clusters'])
        for step in range(self.num['steps']):           
            change = interpolate(x=self.volume_grid['levels'][step],
                                       y=self.stopping_scheme[step],
                                       x_new=volume_cum,
                                       penalty=[0.0, 0.0], mode='straight')
            change = change[:, 0]
            stopping[step, :] = change
            if step < self.num['steps']-1:
                volume_new = volume_cum + change
                #volume_cum = np.dot(transitions[step, :, :].T, volume_new)

                probs = transitions[step, :, :].T*probabilities[step, :]
                probs_sum = probs.sum(axis=1)
                ind_reached = (probs_sum > 0)
                volume_cum = np.dot(probs[ind_reached, :],
                                       volume_new)/probs_sum[ind_reached]


#        a = np.sum(stopping*probabilities, axis=1)
#
#        stopping = np.zeros((self.num['steps'], self.num['clusters']))
#        volume_cum = np.zeros(self.num['clusters'])
#        for step in range(self.num['steps']):
#            change = interpolate(x=self.volume_grid['levels'][step],
#                                       y=self.stopping_scheme[step],
#                                       x_new=volume_cum,
#                                       penalty=[0.0, 0.0], mode='straight')
#            change = change[:, 0]
#            stopping[step, :] = change
#            if step < self.num['steps']-1:
#                volume_new = volume_cum + change
#                #volume_cum = np.dot(transitions[step, :, :].T, volume_new)
#
#                probs = transitions[step, :, :].T*probabilities[step, :]
#                probs_sum = probs.sum(axis=1)
#                ind_reached = (probs_sum > 0)
#                volume_cum = np.dot(probs[ind_reached, :],
#                                       volume_new)/probs_sum[ind_reached]
#
        #b = np.sum(stopping*probabilities, axis=1)
        self.stopping = stopping
        self.delta = np.sum(stopping*probabilities, axis=1)
        self.value['forward'] = np.sum(stopping*self.oqt.centers[:, :, 0]*probabilities)

    def _lsmc_set_process(self, paths):
        # INPUT
        # =====
        # paths    - array with size (num_steps, num_sims, num_factors)

        if not paths.shape[0] == self.num['steps']:
            raise Exception('Given tree does not fit to the set number of steps')
        if not len(paths.shape) > 2:
            paths = paths[:,:,None]
        _, self.num['sims'], self.num['factors'] = paths.shape

        self.paths = paths

    def _lsmc_obtain_stopping_scheme(self):
        grid_size_last = self.volume_grid['size'][-1]
        self.stopping_scheme = self.num['steps']*['']
        values = np.zeros((self.num['sims'], grid_size_last))
        
        for step in range(self.num['steps']-1, -1, -1):
            values_int = interpolate(x=self.volume_grid['levels'][step+1], 
                                     y=values, 
                                     x_new=self.volume_grid['levels'][step],
                                     offset=self.volume_grid['changes'][step])
            if step < self.num['steps']-1:
                values_new = get_expected_value(self.paths[step, :, :], values_int)
            else:
                values_new = np.rollaxis(values_int, axis=1)
            payout = self.paths[step, :, 0, None, None]*self.volume_grid['changes'][step][ :, :, None].T
            values_with_payout = values_new + payout
            best_strategie = np.argmax(values_with_payout, axis=2)
            self.stopping_scheme[step] = self.volume_grid['changes'][step][best_strategie, range(self.volume_grid['size'][step])]
                                                       
            values = np.max(values_with_payout, axis=2) #ToDo: This could be faster, since the maximal element is already obtained two lines above.
        self.value['backward'] = values[:, 0].mean()
        
    def _lsmc_obtain_delta(self):
        stopping = np.zeros((self.num['steps'], self.num['sims']))
        volume_cum = np.zeros(self.num['sims'])
        
        for step in range(self.num['steps']):
            change = interpolate(x=self.volume_grid['levels'][step],
                                       y=self.stopping_scheme[step],
                                       x_new=volume_cum,
                                       penalty=[0.0, 0.0], mode='straight')
            change = change[:, 0]
            #volume_cum_new = volume_cum + change
            #ind_under = volume_cum_new < self.volume_grid['levels'][step+1][0]
            #if ind_under.any():
            #    change[ind_under] = self.volume_grid['levels'][step+1][0] - volume_cum[ind_under]
            volume_cum += change
            stopping[step, :] = change
        self.stopping = stopping
        self.delta = np.mean(stopping, axis=1)        
        self.value['forward'] = np.sum(stopping*self.paths[:, :, 0], axis=0).mean()   

    def _lsmc_obtain_stopping_perfect(self):
        values = np.zeros(self.num['sims']+1)
        volume_top = self.cq['min'] - self.dq['min'].sum()
        volume_flex = self.cq['max'] - self.cq['min']
        dq_span = self.dq['max']-self.dq['min']
        for sim in range(self.num['sims']+1):
            if sim < self.num['sims']:
                path = self.paths[:, sim, 0]
            else:
                path = self.paths[:, :, 0].mean(axis=1)
            ind_sorted = np.argsort(path)[::-1]
             
            stopping_top = np.zeros(self.num['steps'])
            if volume_top > 0:
                dq_cum_sorted = np.cumsum(dq_span[ind_sorted])
                days_top_count = np.where(dq_cum_sorted-volume_top>=0.)[0][0]+1
                days_top = ind_sorted[0:days_top_count]
                stopping_top[days_top] = dq_span[days_top]
                stopping_top[days_top[days_top_count-1]] -= (dq_cum_sorted[days_top_count-1]-volume_top)
            else:
                days_top_count = 0
             
            stopping_flex = np.zeros(self.num['steps'])
            if volume_flex > 0:
                dq_span_flex = dq_span - stopping_top
                dq_flex_cum_sorted = np.cumsum(dq_span_flex[ind_sorted])
                days_flex_count = np.minimum((path > 0).sum(),
                                                np.where(dq_flex_cum_sorted-volume_flex >= 0.)[0][0]+1)
                if days_flex_count > days_top_count:
                    days_flex = ind_sorted[0:days_flex_count]
                    stopping_flex[days_flex] = dq_span_flex[days_flex]
                    if dq_flex_cum_sorted[days_flex_count-1]-volume_flex > 0.:
                        stopping_flex[days_flex[days_flex_count-1]] -= (dq_flex_cum_sorted[days_flex_count-1]-volume_flex)
            stopping = self.dq['min'] + stopping_top + stopping_flex
            values[sim] = np.dot(stopping, path)
        self.value['perfect'] = values[0:self.num['sims']].mean()
        self.value['intrinsic'] = values[self.num['sims']].mean()

