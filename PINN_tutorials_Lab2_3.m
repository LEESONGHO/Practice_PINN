clc; clear all; close all;

%% 주어진 시스템 변수들
m1 = 1; % 질량 m1 (kg)
m2 = 1; % 질량 m2 (kg)
k1 = 5; % 스프링 상수 k1 (N/m)
k2 = 2; % 스프링 상수 k2 (N/m)
L1 = 0.5; % 스프링 1의 자연 길이 (m)
L2 = 0.5; % 스프링 2의 자연 길이 (m)

%% 특성 방정식 설정
syms omega x1(t) x2(t)
eq1 = m1*diff(x1, t, 2) + k1*(x1 - L1) - k2*(x2 - x1 - L2) == 0;
eq2 = m2*diff(x2, t, 2) + k2*(x2 - x1 - L2) == 0;

%% 자연 진동수 ω1과 ω2 계산
[V, D] = eig([k1+k2 -k2; -k2 k2], diag([m1 m2]));
omega_solutions = sqrt(diag(D));

% Omega 계산
omega_1 = omega_solutions(1);
omega_2 = omega_solutions(2);

%% 모드 형상 U 계산
% 각자 모드 형상에 대한 방정식 설정
u11 = V(1,1); % 첫 번째 모드 형상의 첫 번째 요소
u21 = V(2,1); % 첫 번째 모드 형상의 두 번째 요소
u12 = V(1,2); % 두 번째 모드 형상의 첫 번째 요소
u22 = V(2,2); % 두 번째 모드 형상의 두 번째 요소

fprintf('첫 번째 모드 형상: u11 = %f, u21 = %f\n', u11, u21);
fprintf('두 번째 모드 형상: u12 = %f, u22 = %f\n', u12, u22);

%% 초기 조건 설정
x1_0 = 0.5;   % 초기 변위 x1 (m)
x2_0 = 3.25;  % 초기 변위 x2 (m)
dx1dt_0 = 0;  % 초기 속도 dx1dt (m/s)
dx2dt_0 = 0;  % 초기 속도 dx2dt (m/s)


%% Xp 계산
syms K1 K2
eqns = [(k1+k2)*(K1) - (k2)*(K2) == k1*L1 + k2*L2, ...
        -(k2)*(K1) + (k2)*(K2) == k2*L2];
coeff_solutions = solve(eqns, [K1, K2]);
K1_sol = double(coeff_solutions.K1);
K2_sol = double(coeff_solutions.K2);

fprintf('K1_sol: %f\n', K1_sol);
fprintf('K2_sol: %f\n', K2_sol);


%% 연립 방정식 설정: 계수 A, B, C, D 구하기
syms A B C D real
% 초기 조건을 적용한 방정식
eqns = [A*u11 + C*u12 + K1_sol == x1_0, ...
        A*u21 + C*u22 + K2_sol == x2_0, ...
        omega_1*B*u11 + omega_2*D*u12 == dx1dt_0, ... 
        omega_1*B*u21 + omega_2*D*u22 == dx2dt_0]; 

coeff_solutions = solve(eqns, [A, B, C, D]);
A_sol = double(coeff_solutions.A);
B_sol = double(coeff_solutions.B);
C_sol = double(coeff_solutions.C);
D_sol = double(coeff_solutions.D);

% 계수 출력
fprintf('A_sol: %f\n', A_sol);
fprintf('B_sol: %f\n', B_sol);
fprintf('C_sol: %f\n', C_sol);
fprintf('D_sol: %f\n', D_sol);


%% 시간 벡터 설정 및 해 계산
t_max = 5;
numpoints = 250;
t = linspace(0, t_max, numpoints);

% 주어진 해에 따라 x1(t)와 x2(t) 계산
% 주의: 복소수 부분(i)는 MATLAB에서 자동으로 처리하지 않으므로 무시합니다.
% 실제 계산에서는 cos()와 sin() 함수의 계수만 사용합니다.

% 계수 a1, a2는 ω1에 대한 운동을 나타냅니다.
% 계수 a3, a4는 ω2에 대한 운동을 나타냅니다.
x1_t = (A_sol*cos(omega_1*t) + B_sol*sin(omega_1*t)) * u11 + ...
        (C_sol*cos(omega_2*t) + D_sol*sin(omega_2*t)) * u12;

x2_t = (A_sol*cos(omega_1*t) + B_sol*sin(omega_1*t)) * u21 + ...
        (C_sol*cos(omega_2*t) + D_sol*sin(omega_2*t)) * u22;

%% 그래프 그리기
figure;
plot(t, x1_t, 'r', t, x2_t, 'b');
title('Displacement of Masses Over Time');
xlabel('Time (s)');
ylabel('Displacement (m)');
legend('x1(t)', 'x2(t)');
xlim([0 t_max]); % x축 범위 설정
ylim([-5 5]); % y축 범위 설정
grid on;
