import matlab.engine
import numpy as np
import matplotlib.pyplot as plt


class Simulink2Python:
    def __init__(self, modelName='tracking'):
        # The name of the Simulink Model (To be placed in the same directory as the Python Code)
        self.modelName = modelName

    def setControlAction(self, u1, u2):
        # Helper Function to set value of control action
        # here we can try set multiple value in one line, while I failed.
        self.eng.set_param('{}/u1'.format(self.modelName), 'value', str(u1), nargout=0)
        self.eng.set_param('{}/u2'.format(self.modelName), 'value', str(u2), nargout=0)

    def getHistory(self):
        # Helper Function to get system Output and Time History
        out = self.eng.workspace['out']
        return self.eng.eval('out.x1'), self.eng.eval('out.x2'), self.eng.eval('out.tout')

    def connectToMatlab(self):
        print("Starting matlab")
        self.eng = matlab.engine.start_matlab()
        print("Connected to Matlab")

        # Load the model
        self.eng.eval("model = '{}'".format(self.modelName), nargout=0)
        self.eng.eval("load_system(model)", nargout=0)

        # Initialize Control Action to 0
        self.setControlAction(0, 0)
        print("Initialized Model")

        # Start Simulation and then Instantly pause
        self.eng.set_param(self.modelName, 'SimulationCommand', 'start', 'SimulationCommand', 'pause', nargout=0)
        self.x1, self.x2, self.t = self.getHistory()

    def connectController(self, controller):
        self.controller = controller
        self.controller.initialize()

    def simulate(self):
        # Control Loop
        while (self.eng.get_param(self.modelName, 'SimulationStatus') != ('stopped' or 'terminating')):
            # Generate the Control action based on the past outputs
            u1, u2 = self.controller.getControlEffort(self.x1, self.x2, self.t)
            # Set that Control Action
            self.setControlAction(u1, u2)
            # Pause the Simulation for each timestep
            self.eng.set_param(self.modelName, 'SimulationCommand', 'continue', 'SimulationCommand', 'pause', nargout=0)
            self.x1, self.x2, self.t = self.getHistory()

    def disconnect(self):
        self.eng.set_param(self.modelName, 'SimulationCommand', 'stop', nargout=0)
        self.eng.quit()


class Controller:
    def __init__(self):
        # Maintain a History of Variables
        self.x1Hist = []
        self.x2Hist = []
        self.xd1Hist = []
        self.xd2Hist = []
        self.tHist = []
        self.u1Hist = []
        self.u2Hist = []
        self.k = 2  # k is the control gain

    def initialize(self):
        # Initialize the graph
        self.fig1, = plt.plot(self.tHist, self.x1Hist)
        self.fig2, = plt.plot(self.tHist, self.xd1Hist)
        plt.xlim(0, 10)
        plt.ylim(-0.5, 2)
        plt.ylabel("Output")
        plt.xlabel("Time(s)")
        plt.legend('x1', 'xd1', loc='upper right')
        plt.title("System Response")

    def updateGraph(self):
        # Update the Graph
        self.fig1.set_xdata(self.tHist)
        self.fig1.set_ydata(self.x1Hist)
        self.fig2.set_xdata(self.tHist)
        self.fig2.set_ydata(self.xd1Hist)
        plt.ion()
        plt.pause(0.01)
        plt.show()

    def getControlEffort(self, x1, x2, t):
        # Returns control action based on past outputs
        if(type(x1) == float):
            self.x1 = x1
            self.x2 = x2
            self.t = t
        else:
            self.x1 = x1[-1][0]
            self.x2 = x2[-1][0]
            self.t = t[-1][0]

        self.updateGraph()

        # Set Point for x1 and x2 and compute the error terms
        xd1 = np.sin(self.t) + 0.1
        xd2 = -np.cos(0.5 * self.t) - 0.1
        e1 = self.x1 - xd1
        e2 = self.x2 - xd2

        # compute the control inputs
        u1 = -self.k * e1 - self.x2 + np.cos(self.t)
        u2 = -self.k * e2 - self.x1 - self.x2 + 0.5 * np.sin(0.5 * self.t)

        # update histories
        self.x1Hist.append(self.x1)
        self.xd1Hist.append(xd1)
        self.x2Hist.append(self.x2)
        self.xd2Hist.append(xd2)
        self.tHist.append(self.t)
        self.u1Hist.append(u1)
        self.u2Hist.append(u2)
        return u1, u2


tracker = Simulink2Python(modelName="tracking1")
# Establishes a Connection
tracker.connectToMatlab()

# Instantiates the controller
controller = Controller()
tracker.connectController(controller)

# Control Loop
tracker.simulate()

# Closes Connection to MATLAB
tracker.disconnect()
