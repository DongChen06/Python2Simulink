import matlab.engine
import matplotlib.pyplot as plt
import time


class MatEng():
    def __init__(self):
        self.model_address = r'C:\Users\Dong\Google Drive\Dong Chen\Ford_proj\CX482_IVA_PDP_EncryptedSimulinkModel'
        self.modelName = 'Cx482_IVA_forPDP_wDriverModel_realtime_v27_ProtecModel'
        self.eng = None

    def reset_env(self, ):
        self.terminal_state = False
        self.last_reward = 0
        self.t = 0
        self.tHist = []
        self.x1Hist = []
        self.x2Hist = []
        # reuse last engine to save loading time
        if self.eng == None:
            print("Starting matlab")
            self.eng = matlab.engine.start_matlab()
        #else:
            # reset matlab after one epoch
            # self.eng.set_param(self.modelName, 'SimulationCommand', 'stop')
            # self.eng.set_param(self.modelName, 'SimulationCommand', 'start')
            # self.eng.clear('all', nargout=0)

        # go to the model folder
        self.eng.cd(self.model_address, nargout=0)
        # run the simulation configurations (parameters)
        # eng.ls(nargout=0)
        self.eng.Run_Sim(nargout=0)
        # Load the model
        self.eng.eval("model = '{}'".format(self.modelName), nargout=0)
        self.eng.eval("load_system(model)", nargout=0)

        self.setControlAction(0)
        print("Initialized Model")
        # Start Simulation and then Instantly pause
        self.eng.set_param(self.modelName, 'SimulationCommand',
                           'start', 'SimulationCommand', 'pause', nargout=0)
        obs = self.getObservations()
        return obs

    def setControlAction(self,  u1):
        # set value of control action
        self.eng.set_param(
            '{}/Optimal Controller/u1'.format(self.modelName), 'value', str(u1), nargout=0)

    def getObservations(self, ):
        # get system Output and Time History
        tHist = self.eng.eval('tHist')
        v_mph = self.eng.eval('v_mph')
        engine_spd = self.eng.eval('engine_spd')
        MG1_spd = self.eng.eval('MG1_spd')
        MG2_spd = self.eng.eval('MG2_spd')
        Acc_pad = self.eng.eval('Acc_pad')
        Dec_pad = self.eng.eval('Dec_pad')
        WheelTD = self.eng.eval('WheelTD')
        Fuel_kg = self.eng.eval('Fuel_kg')
        SOC_C = self.eng.eval('SOC_C')
        target_speed = self.eng.eval('target_speed')
        eng_ori = self.eng.eval('eng_ori')
        eng_new = self.eng.eval('eng_new')
        if(type(v_mph) == float):
            self.Fuel_kg = Fuel_kg
            self.SOC_C = SOC_C
            self.target_speed = target_speed
            # for plotting use
            self.tHist.append(tHist)
            # self.x1Hist.append(eng_ori)
            # self.x2Hist.append(eng_new)
            self.x1Hist.append(v_mph)
            self.x2Hist.append(target_speed * 0.621371192237334)
            # self.x1Hist.append(int(Fuel_kg) * 1000)
            return (v_mph, engine_spd, MG1_spd, MG2_spd, Acc_pad, Dec_pad, WheelTD)
        else:
            self.Fuel_kg = Fuel_kg[-1][0]
            self.SOC_C = SOC_C[-1][0]
            self.target_speed = target_speed[-1][0]
            # for plotting use
            self.tHist.append(tHist[-1][0])
            # self.x1Hist.append(eng_ori[-1][0])
            # self.x2Hist.append(eng_new[-1][0])  # target_speed[-1][0] * 0.621371192237334
            self.x1Hist.append(v_mph[-1][0])
            self.x2Hist.append(target_speed[-1][0] * 0.621371192237334)
            # self.x1Hist.append(int(Fuel_kg[-1][0]) * 1000)
            return (v_mph[-1][0], engine_spd[-1][0], MG1_spd[-1][0], MG2_spd[-1][0], Acc_pad[-1][0], Dec_pad[-1][0], WheelTD[-1][0])

    def run_step(self, action):
        # u1 = -50 + (action + 1) * 10
        u1 = -200
        # if u1 < 0:
        #     u1 = 0
        # u1 = -10 + (action + 1) * 2
        # Set the Control Action
        self.setControlAction(u1)
        # start = time.time()
        # Pause the Simulation for each timestep
        # self.eng.workspace['Pause_time'] = self.t + 0.3
        self.eng.set_param(self.modelName, 'SimulationCommand',
                           'StepForward', nargout=0)
        # tHist = self.eng.eval('tHist')
        # if type(tHist) == float:
        #     self.t = tHist
        # else:
        #     self.t = tHist[-1][0]
        # print(self.t)
        # end = time.time()
        # print(end - start)

        if (self.eng.get_param(self.modelName, 'SimulationStatus') == ('stopped' or 'terminating')):
            self.terminal_state = True
        # start = time.time()
        obs = self.getObservations()
        # end = time.time()
        # print(end - start)

        # compute the reward
        self.reward_fn()

        # if (self.eng.get_param(self.modelName, 'SimulationStatus') == ('stopped' or 'terminating')):
        #     print(True)
        #     self.terminal_state = True

        return obs, self.last_reward, self.terminal_state, True

    def reward_fn(self,):
        # reward = fuel_consumption + speed_tracking + SOC
        self.last_reward = self.Fuel_kg + self.SOC_C + self.target_speed

    def disconnect(self,):
        print("eng is closed")
        self.eng.set_param(
            self.modelName, 'SimulationCommand', 'stop', nargout=0)
        self.eng.quit()

    def initialize_plot(self, ):
        # Initialize the graph
        self.fig1, = plt.plot(self.tHist, self.x1Hist,
                              color='red', linewidth=1)
        self.fig2, = plt.plot(self.tHist, self.x2Hist, color='k', linewidth=1)
        # for speed tracking
        plt.xlim(0, 800)
        plt.ylim(-10, 150)
        # engine torque
        # plt.xlim(0, 800)
        # plt.ylim(-50, 400)
        # for fuel consumption
        # plt.xlim(0, 800)
        # plt.ylim(0, 3)
        plt.ylabel("Output")
        plt.xlabel("Time(s)")
        # plt.legend('x1', 'x2', loc='upper right')
        plt.title("System Response")

    def updateFig(self, ):
        # Update the Graph
        self.fig1.set_xdata(self.tHist)
        self.fig1.set_ydata(self.x1Hist)
        self.fig2.set_xdata(self.tHist)
        self.fig2.set_ydata(self.x2Hist)
        plt.ion()
        plt.pause(0.001)
        plt.show()
