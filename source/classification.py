__author__ = 'Herakles II'

# imports
import numpy


# routine for starting points
def kmeans(data, start_rule='standard', num_clusters=None, centers=None, miter=1):
    # Classifies given data
    #
    # INPUT
    # =====
    # data          - numpy array with size (self.num['sims'], num_factors)
    # num_clusters  - natural number
    #               > defines the number of different classes
    # start_rule    - '++' or 'standard'
    #               > defines by which algorithm the starting centers are chosen.
    #                 If the argument centers is given ,start_rule will be ignored.
    # centers (Opt.)- numpy array with size (num_factors, num_clusters)
    #               > Defines the used start centroids.
    #
    # OUTPUT
    # ======
    # centers       - array with size (num_clusters, num_factors)
    # mapping       - array with size (num_sims)

    # some parameters
    default_dist = 1e6

    if not len(data.shape) > 1:
        data = data[:, numpy.newaxis]

    num_sims, num_factors = data.shape

    # choose start centers
    if centers is None:
        centers = init_centers(data, num_clusters, method=start_rule)
    else:
        if not len(centers.shape) > 1:
            centers = centers[:, numpy.newaxis]
        num_clusters = centers.shape[0]

    import scipy.cluster.vq as vq
    centers_out, _ = vq.kmeans(data, k_or_guess=centers, iter=miter)
    if centers.shape[0] < num_clusters:
        centers = numpy.zeros_like(centers)
        centers[0:centers.shape[0], :] = centers_out

    distances = default_dist*numpy.ones((num_sims, num_clusters))
    for cluster in range(num_clusters):
        distances[:, cluster] = numpy.sum((data-centers[cluster, :])**2., axis = 1)**(0.5)#lp_norm(data, centers[cluster, :])
    mapping = numpy.argmin(distances, axis=1).astype(numpy.int64)

    return {'centers': centers, 'mapping': mapping}


def lp_norm(x, y, p=2):
    # Calculates lp-norm of the distance x-y
    #
    # INPUT
    # =====
    # x         - numpy vector of size (dim, ) or matrix of size (dim, dim2)
    # y         - numpy vector of size (dim, )
    # p (Opt.)  - positive real number (default = 2)
    #
    # OUPUT
    # =====
    # distance  - real number or vector of size (dim2)

    if len(x.shape) > 1:
        return numpy.sum((x-y[numpy.newaxis, :])**p, axis = 1)**(1.0/p)
    else:
        return numpy.sum((x-y[numpy.newaxis, :])**p)**(1.0/p)

def init_centers(data, num_clusters, method='standard'):
    # Create a set of start centers according to given method.
    #
    # INPUT
    # =====
    # data          - array with size (num_sims, num_factors)
    # num_clusters  - natural number
    # method        - 'standard' or '++'
    #
    # OUTPUT
    # ======
    # centers       - array with size (num_clusters, num_factors)

    num_sims, num_factors = data.shape
    if method == 'standard':
        start_index = numpy.random.choice(range(num_sims), size=num_clusters, replace=False)
        centers = data[start_index, :]
    elif method == '++':
        distance = numpy.inf*numpy.ones(num_sims)
        possible_values = range(num_sims)
        centers = numpy.zeros((num_clusters, num_factors))
        for cluster in range(num_clusters):
            winner_index = numpy.random.choice(possible_values, size=1,
                                               p=distance/distance.sum())
            centers[cluster, :] = data[winner_index[0], :]
            distance = numpy.minimum(distance, lp_norm(data, centers[cluster, :]))
    return centers

def transition_probabilities(mapping):
    # Computes transition probabilities from mapping_from to mapping_to given by mapping.
    # The states correspond to integers included in range(num_clusters).
    # # num_clusters is chosen as max(mapping).
    #
    # INPUT
    # =====
    # mapping        - array with size (num_steps, num_sims)
    #
    # OUTPUT
    # ======
    # transitions    - array with size (num_steps-1, num_clusters_from, num_clusters_to)

    num_steps, num_sims = mapping.shape
    num_clusters = numpy.max(mapping)+1

    transitions = numpy.zeros((num_steps-1, num_clusters, num_clusters))
    for sim in range(num_sims):
        maps = mapping[:, sim]
        index_flat = numpy.ravel_multi_index((range(num_steps-1), maps[0:(num_steps-1)], maps[1:num_steps]), transitions.shape)
        transitions.flat[index_flat] += 1
    rowSum = transitions.sum(axis=2, keepdims=True)
    rowSum[rowSum==0] = 1.
    transitions = transitions/rowSum

    return transitions

class optimal_quantization_tree(object):
    # Creates optimal quantization tree for given data.
        #
        # INPUT
        # =====
        # data          - numpy array with size (num_steps, num_sims, num_factors)
        # num_clusters  - natural number
        #               > defines the number of different classes
        # start_rule    - '++' or 'standard'
        #               > defines by which algorithm the starting centers are chosen.
        #                 If the argument centers is given ,start_rule will be ignored.
        # method        - 'kmeans' or 'cvlq'
        #               > defines method which will be used to find the optimal centroids.
        #
        # OUTPUT
        # ======
        # centers       - array with size (num_steps, num_clusters, num_factors)
        # transitions   - array with size (num_steps-1, num_clusters, num_clusters)
        # probabilities - array with size (num_steps, num_clusters)
    def __init__(self, data=None, start_rule=None, num_clusters=None, method=None, miter=1):

        self.num = {'clusters': num_clusters}
        self.method = method
        self.start_rule = start_rule
        self.miter = miter

        self.data = None
        self.centers = None
        self.transitions = None
        self.probabilities = None

        if method == 'Llyod':
            self.obtain_tree = self._Llyod_obtain_tree
            self.update_tree = self._Llyod_update_tree
        elif method == 'CVLQ':
            self.obtain_tree = self._cvlq_obtain_tree
            self.update_tree = self._cvlq_update_tree
        else:
            raise Exception('Unrecognised method given!')

        self.set_data(data)

    def set_data(self, data=None):
        if data is not None:
            if hasattr(self.num, 'steps'):
                if not self.num['steps'] == data.shape[0]:
                    raise Exception('Given data has wrong size of first dimension. Size must be '+str(self.num['steps']))
            if hasattr(self.num, 'factors'):
                if not self.num['factors'] == data.shape[2]:
                    raise Exception('Given data has wrong size of third dimension. Size must be '+str(self.num['factors']))


            if not len(data.shape)>2:
                data = data[:, :, numpy.newaxis]
            self.data = data
            self.num['steps'], self.num['sims'], self.num['factors'] = data.shape

            if self.centers is None:
                self.centers = numpy.zeros((self.num['steps'], self.num['clusters'], self.num['factors']))


    def _Llyod_obtain_tree(self):
        mapping = numpy.zeros((self.num['steps'], self.num['sims']), dtype=numpy.int_)
        self.probabilities = numpy.zeros((self.num['steps'], self.num['clusters']))

        for step in range(self.num['steps']):
            res = kmeans(self.data[step, :, :], start_rule=self.start_rule,
                         num_clusters=self.num['clusters'])
            self.centers[step, :, :] = res['centers']
            mapping[step, :] = res['mapping']

        self.transitions = transition_probabilities(mapping)

        self.probabilities[0] = numpy.histogram(mapping[0], bins=numpy.arange(self.num['clusters']+1)-0.5)[0]/float(self.num['sims'])
        for step in range(self.num['steps']-1):
            self.probabilities[step+1] = numpy.dot(self.probabilities[step], self.transitions[step])

    def _Llyod_update_tree(self, data):
        self.set_data(data=data)
        self._Llyod_obtain_tree()

    def _cvlq_obtain_tree(self):
        self.transitions = numpy.zeros((self.num['steps']-1, self.num['clusters'], self.num['clusters']))

        for step in range(self.num['steps']):
            self.centers[step, :, :] = init_centers(self.data[step, :, :],
                                                    self.num['clusters'],
                                                    method=self.start_rule)
        self._cvlq_iterate()

    def _cvlq_iterate(self):
        eta = numpy.ones((self.num['steps'], self.num['clusters']))
        for iter in range(self.miter):
            order = numpy.random.permutation(range(self.num['sims']))
            for sim in order:
                path = self.data[:, sim, :]
                diff = self.centers-path[:, None, :]
                distances = numpy.sum(diff**2., axis=2)**(0.5)
                winner = numpy.argmin(distances, axis=1)
                index_flat = numpy.ravel_multi_index((range(self.num['steps']), winner),
                                                         (self.num['steps'], self.num['clusters']))
                eta.flat[index_flat] += 1

                for factor in range(self.num['factors']):
                    self.centers[:,:,factor].flat[index_flat] -= diff[:,:,factor].flat[index_flat]/eta.flat[index_flat]

                self.transitions.flat[numpy.ravel_multi_index((range(self.num['steps']-1),
                                                               winner[0:(self.num['steps']-1)],
                                                               winner[1:self.num['steps']]),
                                                              self.transitions.shape)] += 1
        row_sum = self.transitions.sum(axis=2, keepdims=True)
        row_sum[row_sum == 0] = 1.

        self.transitions = self.transitions/row_sum
        self.probabilities = eta/eta.sum(axis=1, keepdims=True)

    def _cvlq_update_tree(self, data):
        self.set_data(data=data)
        self.transitions = numpy.zeros((self.num['steps']-1, self.num['clusters'], self.num['clusters']))

        self._cvlq_iterate()