clc,
clear all;
% linear system dot_x = A*x+B*u
% dot_x1 = x2+u1
% dot_x2 = x1+x2+u2
% x is the system state, u is the control input
% xd is the reference trajectory
% tracking problem

% initial value of system state
x0 = [1;1];
x = x0;
A = [0 1;1 1];
B = eye(2);

% sampling time 
dt = 0.01;
% simulaiton time
t_total = 10;
% record data
x_history = [];
u_history = [];
xd_history = [];

for t=0:dt:t_total
    % generate reference trajecotry
    xd = [sin(t)+0.1;-cos(0.5*t)-0.1];
    
    % define system error
    e = x-xd;
    
    % control input
    % k is the control gain
    k = 8;
    u = -k*e - A*x + [cos(t);0.5*sin(0.5*t)];
    
    % system update
    dot_x = A*x+B*u;
    x = x+dt*dot_x;
   
    x_history = [x_history x];
    xd_history = [xd_history xd];
    u_history = [u_history u];
end

figure()
hold on;
subplot(2,1,1);
plot(0:dt:t_total,x_history(1,:),0:dt:t_total,xd_history(1,:));
ylabel('x1/xd1');
subplot(2,1,2);
plot(0:dt:t_total,x_history(2,:),0:dt:t_total,xd_history(2,:));
xlabel('time');
ylabel('x2/xd2');
hold off;

figure()
hold on;
subplot(2,1,1);
plot(0:dt:t_total,u_history(1,:));
ylabel('u1');
subplot(2,1,2);
plot(0:dt:t_total,u_history(2,:));
xlabel('time');
ylabel('u2');
hold off;
