import matlab.engine

modelAddress = r'C:\Users\Dong\Google Drive\Dong Chen\Ford_proj\CX482_IVA_PDP_EncryptedSimulinkModel'
modelName = 'Cx482_IVA_forPDP_wDriverModel_realtime_v27_ProtecModel'

eng = matlab.engine.start_matlab()
eng.cd(modelAddress, nargout=0)
# eng.ls(nargout=0)
eng.Run_Sim(nargout=0)
try:
    print("Connected to Matlab")
    eng.eval("model = '{}'".format(modelName), nargout=0)
    eng.eval("load_system(model)", nargout=0)

    eng.set_param('{}/Optimal Controller/u1'.format(modelName),
                  'value', str(0), nargout=0)
    eng.set_param(modelName, 'SimulationCommand', 'start',
                  'SimulationCommand', 'pause', nargout=0)
    print('----')
    v_mph = eng.eval('v_mph')
    if(type(v_mph) == float):
        print(eng.eval('v_mph'))
        print(eng.eval('target_speed'))
        print(eng.eval('Fuel_kg'))
        print(eng.eval('Acc_pad'))
    else:
        target_speed = eng.eval('target_speed')
        print(v_mph[-1][0])
        print(target_speed[-1][0])
        Fuel_kg = eng.eval('Fuel_kg')
        print(Fuel_kg[-1][0])
        Acc_pad = eng.eval('Acc_pad')
        print(Acc_pad[-1][0])
    while (eng.get_param(modelName, 'SimulationStatus') != ('stopped' or 'terminating')):
        eng.set_param('{}/Optimal Controller/u1'.format(modelName),
                      'value', str(0), nargout=0)
        eng.set_param(modelName, 'SimulationCommand', 'continue',
                      'SimulationCommand', 'pause', nargout=0)
        print('----')
        v_mph = eng.eval('v_mph')
        if(type(v_mph) == float):
            print(eng.eval('v_mph'))
            print(eng.eval('target_speed'))
            print(eng.eval('Fuel_kg'))
            print(eng.eval('Acc_pad'))
        else:
            print(1)
            target_speed = eng.eval('target_speed')
            print(v_mph[-1][0])
            print(target_speed[-1][0])
            Fuel_kg = eng.eval('Fuel_kg')
            print(Fuel_kg[-1][0])
            Acc_pad = eng.eval('Acc_pad')
            print(Acc_pad[-1][0])

except Exception as e:
    print("eng is closed")
    eng.set_param(modelName, 'SimulationCommand', 'stop', nargout=0)
    eng.quit()
