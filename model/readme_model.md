## Demonstrative script for the epidemiological model


The script `model.py` demonstrates the features of the epidemiological model used in the paper *Human mobility and sewage data correlate with COVID-19 epidemic evolution in a 3-year surveillance of the metropolitan area of Bologna* by Durazzi, Lunedei, Colombini *et al.*

The example is one year of simulation with a quite strong lockdown 30 days after the arrival of patient zero. The lockdown is later lifted partially, and reinstated twice, with stronger measures after the arrival of a variant at day 290. The variant has relative transmissivity \tau = 1.56. At day 310, a vaccination campaign begins, with 100 doses administered to susceptible individuals each day.

The code here presented simulates the model as presented in the article, encompassing:

- gamma-distributed permanence times in the *Unreported* compartment, within which individuals are able to infect others,

- fixed permanence times for the *Exposed*, *Isolated Infected* and *Recovered*compartments,

- time-dependent transmissivity $\tau(t)$, to model the arrival of new virus variants,

- time-dependent sociability $s(t)$, to model the variations in social habits of the population affected by the epidemic outbreak,

- time-dependent vaccination flow $v(t)$ (expressed for practicality in the code as an absolute flow term $v(t)$ rather than a multiplicative factor $S(t)v(t)$ as indicated in the paper without loss of generality), to implement vaccination campaigns in the model.

The simulated scenario is purely demonstrative and serves the purpose of showcasing the features of the model, and does not reproduce the sequence of events happened during the pandemic in Bologna, the full code originally considered in the article being written in C++, and containing further details related to the organizational features of the local health unit which make it less legible for these purposes.

In addition, the present code is more flexible, as all the delays can be set to be fixed or distributed, and while in this script only gamma and delta distributions are available, one can easily write personalized memory kernels to be used in simulations.

#### Running the example

To run the example you will need the modules `numpy`, `scipy`,`tqdm` and `matplotlib`.

Simply open the current folder in a terminal and run 

`python3 model.py`.
