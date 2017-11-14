__author__ = 'Herakles II'

import copy
import time

import matplotlib.pyplot as plt
import numpy
from simulations import ornstein_uhlenbeck_1d as ou_1d

from source import valuation
from source.classification import optimal_quantization_tree


def test():
    num_steps = 300
    cq_max = 300
    cq_min = 200
    dq_max = 1
    dq_min = 0
    strike = 20.

    num_sims = 1000
    num_clusters = 100
    mrl = 20.0

    swing = valuation.swing(cq_max=cq_max, cq_min=cq_min,
                            dq_max=dq_max, dq_min=dq_min,
                            num_steps=num_steps, relative_spacing=2.,
                            num_actions=2)

    seed = numpy.random.random_integers(1000)
    mu = mrl*numpy.ones(num_steps+1)
    paths = ou_1d(num_steps=num_steps+1, num_sims=num_sims, kappa=0.05,
                  sigma=2, mu=mu, dist_type='arithmetic', seed=seed)
    paths = strike-paths[1:(num_steps+1), :]

    ptm = time.time()
    swing_lsmc = copy.deepcopy(swing)
    swing_lsmc.set_model('lsmc')
    swing_lsmc.set_process(paths)
    swing_lsmc.obtain_stopping_scheme()
    swing_lsmc.obtain_delta()
    #swing_lsmc._lsmc_obtain_stopping_perfect()
    print('Time for pricing with LSMC: ' + str(time.time()-ptm))

    ptm = time.time()
    swing_oqt = copy.deepcopy(swing)
    swing_oqt.set_model('oqt')
    oqt = optimal_quantization_tree(paths, start_rule='standard',
                                    num_clusters=num_clusters, method='CVLQ',
                                    miter=2)
    oqt.obtain_tree()

    swing_oqt.set_process(oqt)
    swing_oqt.obtain_stopping_scheme()
    swing_oqt.obtain_delta()
    print('Time for pricing with oqt: ' + str(time.time()-ptm))

    print('Values:')
    print(swing_oqt.value)
    print('Delta:')
    print(swing_oqt.delta.sum())
    
    print('Values:')
    print(swing_lsmc.value)
    print('Delta:')
    print(swing_lsmc.delta.sum())

    t = range(num_steps)
    plt.plot(t, swing_lsmc.delta, t, swing_oqt.delta)#, t, delta_num_oqt)
    plt.show()

    centers = oqt.centers[:, :, 0]
    data = paths

    #mapping = numpy.zeros((num_steps, num_sims))
    #for step in range(num_steps):
    #    distances = 1000000*numpy.ones((num_sims, num_clusters))
    #    for cluster in range(num_clusters):
    #        distances[:, cluster] = numpy.abs(data[step, :]-centers[step, cluster])
    #    mapping[step, :] = numpy.argmin(distances, axis=1).astype(numpy.int64)

    #stopping_mean = numpy.zeros((num_steps, num_clusters))
    #for step in range(num_steps):
    #    for center in range(num_clusters):
    #        ind = mapping[step, :] == center
    #        stopping_mean[step, center] = swing_lsmc.stopping[step, ind].mean()
    #
    #a = 1

def testProfile():
    import cProfile

    command = """test()"""

    cProfile.runctx( command, globals(), locals(), filename="test.profile" )
    
if __name__ == '__main__':
    #testProfile()
    test()
    
#ToDos
#1. Better interpolation e.g. splines? -> Way too slow
#2. Interpolation without loop? -> almost there
#3. When getting pathwise stopping, can one figure out where to interpolate first? -> done
#4. Why is oqt so slow? -> done
#5. LSMC? -> done
#6. Faster perfect stopping??
#8. Test Gamma-Pos