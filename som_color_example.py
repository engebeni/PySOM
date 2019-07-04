import matplotlib.pyplot as pl
import numpy as np
import time
from som import SOM

#the samples with which the map will be trained, basically these are vectors representing RGB colors
training_samples = np.array(
         [[0., 0., 0.],
          [0., 0., 1.],
          [0., 0., 0.5],
          [0.125, 0.529, 1.0],
          [0.33, 0.4, 0.67],
          [0.6, 0.5, 1.0],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 0., 1.],
          [1., 1., 0.],
          [1., 1., 1.],
          [.33, .33, .33],
          [.5, .5, .5],
          [.66, .66, .66]])
         
#the color's corresponding names, to label the plot        
color_names = \
	['black', 'blue', 'darkblue', 'skyblue',
	'greyblue', 'lilac', 'green', 'red',
	'cyan', 'violet', 'yellow', 'white',
	'darkgrey', 'mediumgrey', 'lightgrey']

som = SOM(60, 100, 3)
start = time.time()
scaling_factor = 1e2
som.train(training_samples, 400, 0.8, scaling_factor, 20, scaling_factor)
end = time.time()
total_time = end - start
print("Elapsed time: {}".format(total_time))
pl.imshow(som.som, origin='lower')

#label each weight with its corresponding name
for i in range(len(training_samples)):
    bmu = som.get_bmu(training_samples[i])
    pl.text(bmu[1], bmu[0], color_names[i], ha='center', va='center',
           bbox=dict(facecolor='white', alpha=0.5, lw=0))

pl.show()