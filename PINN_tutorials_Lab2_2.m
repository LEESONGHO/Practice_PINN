clc; clear all; close all; 

%%
syms omega A B C D

m1 = 1; % 질량 m1 (kg)
m2 = 1; % 질량 m2 (kg)
k1 = 5; % 스프링 상수 k1 (N/m)
k2 = 2; % 스프링 상수 k2 (N/m)

% 특성 방정식의 행렬
M = [m1 0; 0 m2];
K = [k1+k2 -k2; -k2 k2];
char_eq = det(-omega^2*M + K);

% 자연 진동수 구하기 (omega를 풀이)
omega_solutions = solve(char_eq, omega);

% 특성 방정식의 해를 구합니다 (자연 진동수를 구합니다).
omega_solutions = double(solve(char_eq, omega, 'Real', true));

% 초기 조건
x1_0 = 0.5;   % 초기 변위 x1 (m)
x2_0 = 3.25;     % 초기 변위 x2 (m)
dx1dt_0 = 0; % 초기 속도 dx1dt (m/s)
dx2dt_0 = 0;    % 초기 속도 dx2dt (m/s)

%%
syms A B C D % real;

% % 연립 방정식 설정
% eqns = [A*cos(omega_solutions(1)*0) + B*sin(omega_solutions(1)*0) == x1_0, ...
%         -A*omega_solutions(1)*sin(omega_solutions(1)*0) + B*omega_solutions(1)*cos(omega_solutions(1)*0) == dx1dt_0, ...
%         C*cos(omega_solutions(2)*0) + D*sin(omega_solutions(2)*0) == x2_0, ...
%         -C*omega_solutions(2)*sin(omega_solutions(2)*0) + D*omega_solutions(2)*cos(omega_solutions(2)*0) == dx2dt_0];

%%
omega_1 = omega_solutions(1);
omega_2 = omega_solutions(3);

% 초기 조건
x1_0 = 0.5;
x2_0 = 3.25;
dx1dt_0 = 0;
dx2dt_0 = 0;

% A, B, C, D에 대한 연립 방정식 설정
% x1(t) = A*cos(ω1*t) + B*sin(ω1*t)
% x2(t) = C*cos(ω2*t) + D*sin(ω2*t)
% 초기 조건을 사용해 A, B, C, D를 구합니다.
eqns = [A == x1_0, ...
        C == x2_0, ...
        omega_1*B == dx1dt_0, ...
        omega_2*D == dx2dt_0];

% 연립 방정식을 풀어서 계수를 구합니다.
coeff_solutions = solve(eqns, [A, B, C, D]);

% 각 계수의 해를 double로 변환합니다.
A_sol = double(coeff_solutions.A);
B_sol = double(coeff_solutions.B);
C_sol = double(coeff_solutions.C);
D_sol = double(coeff_solutions.D);

% 계수 출력
fprintf('계수 A: %f\n', A_sol);
fprintf('계수 B: %f\n', B_sol);
fprintf('계수 C: %f\n', C_sol);
fprintf('계수 D: %f\n', D_sol);

%%
% 시간 벡터 설정
t_max = 5;
numpoints = 250;
t = linspace(0, t_max, numpoints);

% x1(t)와 x2(t)를 계산합니다.
x1_t = A_sol * cos(omega_1*t) + B_sol * sin(omega_1*t);
x2_t = C_sol * cos(omega_2*t) + D_sol * sin(omega_2*t);

% 그래프를 그립니다.
figure;
plot(t, x1_t, 'r', t, x2_t, 'b');
title('Displacement of Masses Over Time');
xlabel('Time (s)');
ylabel('Displacement (m)');
legend('x1(t)', 'x2(t)');
xlim([0 5]); % y축 범위 설정
ylim([-5 5]); % y축 범위 설정
grid on;




