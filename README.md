# distributed-sensing

A repository to support the paper **Spatial Scheduling of Informative Meetings for Multi-Agent Persistent Coverage**.

Videos:  
[![click to play](https://img.youtube.com/vi/M5Fp8WsmLno/0.jpg)](https://www.youtube.com/watch?v=M5Fp8WsmLno)
[![click to play](https://img.youtube.com/vi/gLdqK2m3COo/0.jpg)](https://www.youtube.com/watch?v=gLdqK2m3COo)

Paper citation:
```
@Article{9001230,
  author={R. N. {Haksar} and S. {Trimpe} and M. {Schwager}},
  journal={IEEE Robotics and Automation Letters}, 
  title={Spatial Scheduling of Informative Meetings for Multi-Agent Persistent Coverage}, 
  year={2020},
  volume={5},
  number={2},
  pages={3027-3034},}
```

## Requirements:
- Developed with Python 3.6
- Requires the `numpy` and `networkx` packages
- Requires the [simulators](https://github.com/rhaksar/simulators) repository 

## Directories:
- `framework`: Implementation of scheduling and planning framework. 

## Files:
- `Baseline.py`: Implementation of two baseline algorithms to compare against the framework. 
- `Benchmark.py`: Run many simulations of the framework to evaluate perofrmance. 
- `Meetings.py`: Run a single simulation of the framework. 
