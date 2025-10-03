# **nuSYD: A simple method to measure numax for asteroseismology.**

This is the python implementation of the oscillation detection algorithm proposed by [Sreenivas et al. 2024](https://academic.oup.com/mnras/article/530/3/3477/7643660?login=true). This algorithm is fast, simple and easy to implement.

```python

from nusyd import nuSYD

time =    #your time array
flux =    #your flux array

runner =  nuSYD(time, flux, mc_iter = 100)

results = runner.run()

print("Your numax = {:.4f} +/- {:.4f} $\mu$Hz".format(results["numax"], results["errors"])

```
Set ``` mc_iter = False (by default)```, if you only need numax.

The algorithm is good in detecting oscillations without an initial guess, in most cases :-)

Check todo list for future works!!!

Contact me at : skal9597@uni.sydney.edu.au for an suggestions.


