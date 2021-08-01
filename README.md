#  Double Inverted Pendulum Swing-Up and Stabilization using MPC 
This project aims to control a double inverted pendulumusing model predictive control (MPC). The starting position is set to the origin of coordinates with both pendulums hanging down. The MPC approach shall be able to, within physical limitations, bring the cart and pendulums in any arbitrary position while avoiding given obstacles.
For the problem at hand a direct MPC approach utilizing orthogonal collocation is used. Additionaly a cost function is proposed that includes a quadratic set-point tracking objective. The algorithm developed is documented in a Jupyter Notebook according to the task requirements.

### Results


### Installation  
  1. [Download and install Conda](https://conda.io/docs/download.html)
     * Install conda enviroment:
     
     	* Ubuntu: 
            ```
            conda env create -f environment.yml
            ```
     		Then:
     		```
     		source activate mpc_env  
     		```
     	
        * Windows:
     		```
     		conda env create -f environment.yml
     		```
     	Then:
     		```
            conda activate mpc_env
     		```
