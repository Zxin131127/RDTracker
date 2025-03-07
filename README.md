# Regularized-Aware Discriminative Transformer Tracker for Satellite Videos
[Xin Zhang](https://ieeexplore.ieee.org/author/37086137172), [Licheng Jiao](https://ieeexplore.ieee.org/author/37276095000), [Lingling Li](https://ieeexplore.ieee.org/author/37085521934), [Xiaoyan Yang](https://ieeexplore.ieee.org/author/480730827513052), [Zhongjian Huang](https://ieeexplore.ieee.org/author/37087039303), [Xu Liu](https://ieeexplore.ieee.org/author/37367391400), [Fang Liu](https://ieeexplore.ieee.org/author/37401762200), [Wenping Ma](https://ieeexplore.ieee.org/author/37575202300) and [Shuyuan Yang](https://ieeexplore.ieee.org/author/37279517100)

## Tracking Results

**Tracking results:** the raw results of RDTracker on 3 benchmarks including SV248S, VISO, and SatSOT can be found in Results dir.

## Environment Setup

#### Clone the GIT repository.  
```bash
git clone https://github.com/Zxin131127/RDTracker.git
```
#### Clone the submodules.  
In the repository directory, run the commands:  
```bash
git submodule update --init  
```  
#### Install dependencies
Run the installation script to install all the dependencies. You need to provide the conda install path (e.g. ~/anaconda3) and the name for the created conda environment (here ```rdtracker```).  
```bash
bash install.sh conda_install_path rdtracker
```  
This script will also download the default networks and set-up the environment.  

**Note:** The install script has been tested on an Ubuntu 20.04 system. In case of issues, check the [detailed installation instructions](INSTALL.md). 

## Testing the RDTracker

```bash
python pytracking/run_experiment.py result_path net_path trdimp_trackingnet cascade_level
```

## Contact
If you have any questions, please feel free to contact zhangxin131127@163.com

