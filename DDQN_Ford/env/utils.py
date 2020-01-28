import matlab.engine
import matplotlib.pyplot as plt


def connectToMatlab(eng, modelName):
    print("Connected to Matlab")
    # Load the model
    eng.eval("model = '{}'".format(modelName), nargout=0)
    eng.eval("load_system(model)", nargout=0)


def reset_env(eng, modelName):
    # Initialize Control Action to 0
    setControlAction(eng, 0, modelName)
    print("Initialized Model")

    # Start Simulation and then Instantly pause
    eng.set_param(modelName, 'SimulationCommand', 'start', 'SimulationCommand', 'pause', nargout=0)
    obs = getObservations(eng)
    return obs

def setControlAction(eng, u1, modelName):
    # Helper Function to set value of control action
    eng.set_param('{}/u1'.format(modelName), 'value', str(u1), nargout=0)


def getObservations(eng, ):
    # Helper Function to get system Output and Time History
    return (eng.eval('v_mph'), eng.eval('engine_spd'), eng.eval('MG1_spd'),
           eng.eval('MG2_spd'), eng.eval('Acc_pad'), eng.eval('Dec_pad'),
           eng.eval('WheelTD'), eng.eval('Fuel_kg'), eng.eval('SOC_C'),
            eng.eval('target_speed'))

def run_step(eng, modelName, u1, u2):
    # Control Loop
    while (eng.get_param(modelName, 'SimulationStatus') != ('stopped' or 'terminating')):
        # Set that Control Action
        setControlAction(eng, u1, u2, modelName)
        # Pause the Simulation for each timestep
        eng.set_param(modelName, 'SimulationCommand', 'continue', 'SimulationCommand', 'pause', nargout=0)
        obs = getObservations(eng, )
    return obs


def reward_fn():
    pass


def disconnect(eng, modelName):
    eng.set_param(modelName, 'SimulationCommand', 'stop', nargout=0)
    eng.quit()


def initialize_plot(tHist, x1Hist, xd1Hist):
    # Initialize the graph
    fig1, = plt.plot(tHist, x1Hist)
    fig2, = plt.plot(tHist, xd1Hist)
    plt.xlim(0, 10)
    plt.ylim(-0.5, 2)
    plt.ylabel("Output")
    plt.xlabel("Time(s)")
    plt.legend('x1', 'xd1', loc='upper right')
    plt.title("System Response")
    return fig1, fig2


def updateFig(fig1, fig2, tHist, x1Hist, xd1Hist):
    # Update the Graph
    fig1.set_xdata(tHist)
    fig1.set_ydata(x1Hist)
    fig2.set_xdata(tHist)
    fig2.set_ydata(xd1Hist)
    plt.ion()
    plt.pause(0.01)
    plt.show()
