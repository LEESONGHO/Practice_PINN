clc; clear all; close all; 

%%
% 주어진 시스템 변수
m = 1;  % 질량 (kg)
k = 5;  % 스프링 상수 (N/m)

% 자연 진동수 (rad/s)
omega_n = sqrt(k / m);

% 초기 조건
x0 = 0.5;     % 초기 변위 (m)
v0 = 0;       % 초기 속도 (m/s)

% 계수 A2는 초기 변위와 동일
A2 = x0;

% 계수 A1을 구하기 위한 초기 속도 조건 이용
% v(t) = A1*omega_n*cos(omega_n*t) - A2*omega_n*sin(omega_n*t)
% 초기 속도 v0는 t=0에서의 속도이므로,
A1 = v0 / omega_n;

% 계수 출력
fprintf('계수 A1: %f\n', A1);
fprintf('계수 A2: %f\n', A2);
