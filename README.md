# revpref

Python Tools for Computational Revealed Preference Analysis

## Synopsis

Our package provides a set of tools to

i) check consistency of a finite set of consumer demand observations with GARP

ii) compute goodness-of-fit indices when the data do not obey the axioms

We implement various method to compute following indices:

### CCEI (Critical Cost Efficiency index, *Afrait 1972*)

* CCEI (also known as the Afriat efficiency index). The CCEI is defined as the maximal value of the efficiency level e at which the data set is consistent with GARP.
* Generally, A is directly prefered to B when $p^Aq^A>p^Aq^B$, and CCEI is the suprema of e under constraint $e p^Aq^A>p^Aq^B$ to satisfy GARP .
* We provide three method to compute CCEI. (**Warshall**, MTZ and bisection)

### HMI (Houtman and Maks index, *Houtman and Maks, 1985*)

* The Houtman-Maks index gives the relative size of the largest subset of observations that is still consistent with GARP.
* For instance, if we delete the tenth choice from a 10-choice dataset and find it satisfy GARP, then the HMI is (10 - 1)/10 = 0.9
* We provide two methods to compute CCEI. (GrossKaiser, **MTZ**)

### AVI (Average Afrait index, *Varian 1990*)

* AVI thinks that every choice has its own e. AVI is the maxium of the mean of e-vector.
* We provide only one method to compute AVI. (**MTZ**)

### MPI (Money Pump index, *Echenique et al., 2011*)

* The MPI is the amount of money one can extract from a consumer who violates the axioms.
* We provide a general algorithm to compute MPI under finite max length of cycles, and we also provide a faster matrix algorithm for max length = 2. (Cycle, **Matrix**)

### MCI (Minimal Cost index, *Dean and Martin, 2016*)

* MCI is defined as the minimum cost of removing all cycles from the graph.
* We provide two methods to compute MCI. (Optimize, **MTZ**)

## Installation

You can install our package from [PyPI](https://pypi.org/) using the following command

```shell
pip install revpref
```

or from github and install it manually

```shell
git clone https://github.com/Aliebc/revpref.git
cd revpref
python3 setup.py install
```

## Requirements

Python >= 3.10

numpy, scipy, networkx and pulp are required.

## Example

Below we provide some simple examples to illustrate the main class in our package.

```python
import revpref as rp
import numpy as np

p = np.array(
    [[4, 4, 4], 
     [1, 9, 3], 
     [2, 8, 3], 
     [1, 8, 4], 
     [3, 1, 9], 
     [3, 2, 8], 
     [8, 4, 1], 
     [4, 1, 8], 
     [9, 3, 1], 
     [8, 3, 2]]
)

q = np.array(
    [[1.81, 0.19, 10.51], 
     [17.28, 2.26, 4.13], 
     [12.33, 2.05, 2.99], 
     [6.06, 5.19, 0.62], 
     [11.34, 10.33, 0.63], 
     [4.33, 8.08, 2.61], 
     [4.36, 1.34, 9.76], 
     [1.37, 36.35, 1.02], 
     [3.21, 4.97, 6.20], 
     [0.32, 8.53, 10.92]]
)

pref = rp.RevealedPreference(p, q) # Construct the class
print(pref.check_garp()) # Check if satisfy GARP

print("---CCEI---")
print(pref.ccei()) # Compute the CCEI
print(pref.ccei(method='bisection', tol = 1e-10))
print(pref.ccei(method='mtz')) # Use other methods

print("---AVI---")
print(pref.avi())

print("---MPI---")
print(pref.mpi())
print(pref.mpi(method='matrix'))

print("---MCI---")
print(pref.mci())
print(pref.mci(method='mtz'))
# Accelerate if you have Gurobi Optimizer or CPLEX
# print(pref.mci(method='mtz', lp_solver = 'GUROBI_CMD')) 

print("---HMI---")
print(pref.hmi())
print(pref.hmi(method='mtz'))
```

Expected output:

```python
False
---CCEI---
0.9488409272581934
0.9488409272162244
0.94884093
---AVI---
0.9948840929999999
---MPI---
0.0724275724275725
0.0724275724275725
---MCI---
0.0051182597916708365
0.0051182597916708365
---HMI---
0.9
0.9
```

## References

[1]  Afriat, Sidney N."Efficiency estimation of production functions." *International economic review* (1972): 568-598.

[2]  Houtman, Martijn, and Julian Maks. "Determining all maximal data subsets consistent with revealed preference." *Kwantitatieve methoden* 19, no. 1 (1985): 89-104.

[3]  Varian, Hal R. "Goodness-of-fit in optimizing models." *Journal of Econometrics* 46, no. 1-2 (1990): 125-140.

[4]  Echenique, Federico, Sangmok Lee, and Matthew Shum. "The money pump as a measure of revealed preference violations." *Journal of Political Economy* 119, no. 6 (2011): 1201-1223.

[5]  Dean, Mark, and Daniel Martin. "Measuring rationality with the minimum cost of revealed preference violations." *Review of Economics and Statistics* 98, no. 3 (2016): 524-534.

[6]  Demuynck, Thomas, and John Rehbeck. "Computing revealed preference goodness-of-fit measures with integer programming." *Economic Theory* 76, no. 4 (2023): 1175-1195.

[7]  Gross, John, and Dan Kaiser. "Two simple algorithms for generating a subset of data consistent with warp and other binary relations." *Journal of Business & Economic Statistics* 14, no. 2 (1996): 251-255.
