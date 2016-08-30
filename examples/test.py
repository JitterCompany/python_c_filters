import numpy as np

import cfilt

a = np.array([[1.2, 2.3, 2.4, 1.0, 0.9, 0.8], [2.3, -2.1, 1.1, 1.0, -1.2, 3.1]], dtype='float64')

filterIndex = cfilt.filter64_init(a)

#x = np.array([2.1,2.1,3.1,4.1], dtype='float64')
x = np.zeros(220, dtype='float64')

y = cfilt.filter64_apply(filterIndex, x)

print(y)
