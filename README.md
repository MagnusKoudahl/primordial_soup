# Primordial Soup Kitchen!
Experiments with recreating the Primordial Soup from Friston (2019), A Free Energy Principle for a Particular Physics

soup.jl and functional_soup.jl are the files that run the simulation. The rest will get cleaned up. This repo will probably also recreate "Life as We Know It" in the future.

Prior to convergence (~t=1000) some particles will launch themselves far away. This happens when 2 particles are forced extremely close. The $\Delta_{ij}^2$ factor blows up, causing an extremely large repulsive force. Not sure if this is a bug or a feature?
