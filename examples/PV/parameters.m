% inverter filter calculation

Tss = 2.5e-6;  % sampling time

P = 10e3;  % rated active power
U = 380;  % inverter phase2phases voltage
f= 50;  % frequency
fsw = 5e3;  % swicthing frequency

Cfmax = (0.05*P) / (2*pi*f*U^2);  % = 11uF

Lf = (0.1*U^2)/(2*pi*f*P); % 4.6mH
RLf = Lf*100; %resistance of inductor