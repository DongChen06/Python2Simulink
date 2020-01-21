Python2Simulink
===============
#### Robotics and Intelligent Vehicle Automation Lab (RIVAL)
- Built by Dong Chen
- Started on Jan.11, 2020

A bridge between Python and Simulink.
This file aims to build a bridge between Python and Simulink. At each time step, the python script will send a command (input) to the simulink model, then the simulink model executes for one step and then returns the results to Python scripts for the decision usage. 

## Install MATLAB Engine API for Python
-------

Install the MATLAB Engine API follow the instruction [Installation](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

## Applications
-------

1. plant example

In this example, we build a PI controller to regulate a secord-order system to a reference value (10 here). The Python script compute the control input and sends then value to the Simulink model. Then Simulink model runs for one step and returns the output value to the python script. 

<p align="center">
     <img src="Docs/plant.gif" alt="output_example" width="60%" height="60%">
     <br>Fig.1 Figure of regulation result
</p>

2. tracking example

In this example, the controller tries to control the variables x1 and x2 to a regulate track xd1 and xd2. There is also a code-like model and controller built by [Matlab code](examples/tracking/linear_controller.m).

The simulink model looks like this,

<p align="center">
     <img src="Docs/simulink_model.png" alt="output_example" width="60%" height="60%">
     <br>Fig.2 simulink model
</p>

where the u1 and u2 are control input from the Python script. The state variables x1 and x2 will be sent to the python script. 

<p align="center">
     <img src="Docs/tracking.gif" alt="output_example" width="60%" height="60%">
     <br>Fig.3 Tracking result, x1(blue) and xd1(orange)
</p>

## Gym-like Reinforcement Learning interface combined with Python and Matlab

In progress...
 
 
## API
-------

- start the engine and connect to Matlab; Load the model:

```
matlab.engine.start_matlab()
self.eng.eval("model = '{}'".format(self.modelName),nargout=0)
self.eng.eval("load_system(model)",nargout=0)
 ```

- eng.eval('out.output')

- eng.workspace()

- self.eng.set_param('{}/u'.format(self.modelName),'value',str(u),nargout=0): set the control input u.

- Start Simulation and then Instantly pause and read output.

```
eng.set_param(self.modelName,'SimulationCommand','start','SimulationCommand','pause',nargout=0)
out = self.eng.workspace['out']
self.eng.eval('out.output'), self.eng.eval('out.tout')
```

- set_param

learn how to set parameters of blocks [check here](https://www.mathworks.com/help/simulink/slref/set_param.html)

## Reference:
-------

1. [link1](https://stackoverflow.com/questions/48864281/executing-step-by-step-a-simulink-model-from-python)
2. [Calling MATLAB from Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html)
3. [Troubleshoot MATLAB Errors in Python](https://www.mathworks.com/help/matlab/matlab_external/troubleshoot-matlab-errors-in-python.html)
