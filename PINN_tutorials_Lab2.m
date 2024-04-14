clc; clear all; close all; 

%%
% Given system properties
m1 = 1;  % kg
m2 = 1;  % kg
k1 = 5;  % N/m
k2 = 2;  % N/m
L1 = 0.5; % m
L2 = 0.5; % m

% Initial conditions
init_cond = [0.5; 0; 3.25; 0];

% Time span
tspan = [0 10];  % simulate for 10 seconds

% Solve ODE
[t, y] = ode45(@(t, y) mass_spring_ode(t, y, m1, m2, k1, k2, L1, L2), tspan, init_cond);

% Plot results
figure;
plot(t, y(:,1), 'r', t, y(:,3), 'b');
title('Displacement of Masses Over Time');
xlabel('Time (s)');
ylabel('Displacement (m)');
legend('m1', 'm2');

%%
function dydt = mass_spring_ode(t, y, m1, m2, k1, k2, L1, L2)
    % Unpack the current state
    x1 = y(1);
    v1 = y(2);
    x2 = y(3);
    v2 = y(4);

    % Differential equations
    dx1dt = v1;
    dv1dt = (k2*(x2 - x1 - L2) - k1*(x1 - L1)) / m1;
    dx2dt = v2;
    dv2dt = (-k2*(x2 - x1 - L2)) / m2;

    % Output derivative
    dydt = [dx1dt; dv1dt; dx2dt; dv2dt];
end