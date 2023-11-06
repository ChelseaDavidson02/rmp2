# Online Surface Following Algorthm using Riemannian Motion Policies
Code adaped from the RMP2 code to enable surface following in a dynamic environment. The README for the original code can be found at the bottom of this file.

### Installation 
Download the rmp2 zip file or run:
```
git clone https://github.com/ChelseaDavidson02/rmp2.git
```

### System integration
This code operates in Ubuntu and must therefore be run on an Ubuntu device or in WSL on Windows Devices.

Ensure that a virtual environment manager is installed. The following code assumes anaconda3 is used.

Setup a virtual environment by running:
```
conda env create -f environment.yml
```

To activate the environment and export the python path, run:
```
. startup.sh
```

### Surface following code
To run the surface following algorithm in the simulated environment, run:
```
python examples/rmp2/surface_following.py 
```


# RMP2


Code for R:SS 2021 paper *RMP2: A Structured Composable Policy Class for Robot Learning*. [[Paper](https://arxiv.org/abs/2103.05922)] 

### Installation
```
git clone https://github.com/UWRobotLearning/rmp2.git
cd rmp2
conda env create -f environment.yml
. startup.sh
```

### Hand-designed RMP2 for Robot Control
To run a goal reaching task for a Franka robot:
```
python examples/rmp2/rmp2_franka.py
```

To run a goal reaching task for a 3-link robot:
```
python examples/rmp2/rmp2_3link.py
```

### Training RMP2 Policies with RL
**Note:** The instruction below is for the 3-link robot. To run experiments with the franka robot, simply replace `3link` by `franka`.

To train an NN policy from scratch (without RMP2):
```
python run_3link_nn.py
```

To train an NN residual policy:
```
python run_3link_nn.py --env 3link_residual
```

To train an RMP residual policy:
```
python run_3link_residual_rmp.py
```

To restore training of a policy:
```
python restore_training.py --ckpt-path ~/ray_results/[EXPERIMENT_NAME]/[RUN_NAME]/
```

To visualize the trained policy:
```
python examples/rl/run_policy_rollouts.py --ckpt-path ~/ray_results/[EXPERIMENT_NAME]/[RUN_NAME]/
```

### Citation
If you use this source code, please cite the below article,

```
@inproceedings{Li-RSS-21,
    author = "Li, Anqi and Cheng, Ching-An and Rana, M Asif and Xie, Man and Van Wyk, Karl and Ratliff, Nathan and Boots, Byron",
    booktitle = "Robotics: Science and Systems ({R:SS})",
    title = "{{RMP}2: A Structured Composable Policy Class for Robot Learning}",
    year = "2021"
}
```
