# Python2Simulink
A bridge between Python and Simulink
This file aims to Call Simulink module with Python.

## Install MATLAB Engine API for Python

Install the MATLAB Engine API follow the instruction [Installation](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).


## API
- matlab.engine.start_matlab():
start the engine

- eng.eval("model = '{}'".format(self.modelName),nargout=0)

- eng.eval('out.output')

- eng.workspace()

- eng.set_param(self.modelName,'SimulationCommand','start','SimulationCommand','pause',nargout=0)



## Reference:
1. [link1](https://stackoverflow.com/questions/48864281/executing-step-by-step-a-simulink-model-from-python)
2. [Calling MATLAB from Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html)
3. [Troubleshoot MATLAB Errors in Python](https://www.mathworks.com/help/matlab/matlab_external/troubleshoot-matlab-errors-in-python.html)
