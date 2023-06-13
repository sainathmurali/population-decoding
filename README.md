This python code is for the population decoding of task variables from calcium fluorescence (df/f) data of mice performing whisker-based tactile discrimination task recorded in multiple brain areas. Some of the details of the code are as follows:
  * The classifiers used here were Support Vector Classifier (SVC) and Linear Discriminant Analysis (LDA) from scikitlearn.
  * Moving (chunk with no overlap) window (or) cumulative (time points added cumulatively to the chunk) window was used to capture the changes in fluorescence over time.

The details of the task, recording methods and areas can be found in this paper (https://doi.org/10.1016/j.cub.2020.10.059).
The data and the code of the single-neuron level analysis done in the paper can be found here (https://doi.org/10.25377/sussex.12573881).
