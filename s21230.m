clc; clear; close all;

%% =====================================================
%  Fixed-time Bipartite Consensus with ESO
%  (Paper-level simulation)
%% =====================================================

%% ---------- Basic parameters ----------
N  = 6;          % number of followers
T  = 10;         % prescribed time
Ts = 0.5;        % time-scaling parameter
dt = 1e-3;
t  = 0:dt:T-dt;

%% ---------- Signed communication graph ----------
A = [ 0  1  0  0  0  0;
      1  0 -1  0  0  0;
      0 -1  0  1  0  0;
      0  0  1  0 -1  0;
      0  0  0 -1  0  1;
      0  0  0  0  1  0];

D = diag([1 -1 1 -1 1 -1]);      % structural balance matrix
L = diag(sum(abs(A),2)) - A;

%% ---------- Control gains ----------
k1 = 2.0;
k2 = 4.0;
k3 = 4.0;
k4 = 2.0;

%% ---------- ESO gains ----------
a1 = 10;
a2 = 15;
a3 = 20;

%% ---------- Initial conditions ----------
x0 = 1.5;     v0 = 0.5;          % leader
x  = randn(N,1);
v  = randn(N,1);

xh = zeros(N,1);                 % ESO states
vh = zeros(N,1);
wh = zeros(N,1);

%% ---------- Data storage ----------
X  = zeros(N,length(t));
V  = zeros(N,length(t));
X0 = zeros(1,length(t));

%% =====================================================
%                Main simulation loop
%% =====================================================
for k = 1:length(t)

    tk = t(k);
    gamma = T/(T - tk + Ts);

    %% ---- Leader dynamics ----
    f0 = 0.5*sin(x0);            % unknown nonlinear term
    x0 = x0 + dt*v0;
    v0 = v0 + dt*f0;

    %% ---- Followers ----
    for i = 1:N

        % Bipartite consensus errors
        e1 = x(i) - D(i,i)*xh(i);
        e2 = v(i) - D(i,i)*vh(i);

        % Control law
        u = -k1*e1 ...
            -k2*gamma*e1 ...
            -k3*(1/gamma)*e2 ...
            -k4*e2 ...
            + D(i,i)*wh(i);

        % External disturbance
        w = 0.3*sin(2*tk + i);

        % System update
        x(i) = x(i) + dt*v(i);
        v(i) = v(i) + dt*(u + w);

        %% ---- Distributed ESO ----
        sum_x = 0; sum_v = 0; sum_w = 0;
        for j = 1:N
            sum_x = sum_x + A(i,j)*(xh(j) - xh(i));
            sum_v = sum_v + A(i,j)*(vh(j) - vh(i));
            sum_w = sum_w + A(i,j)*(wh(j) - wh(i));
        end

        xh(i) = xh(i) + dt*( ...
            vh(i) ...
            - a1*gamma*(xh(i) - x0) ...
            + sum_x );

        vh(i) = vh(i) + dt*( ...
            wh(i) + u ...
            - a2*gamma*(vh(i) - v0) ...
            + sum_v );

        wh(i) = wh(i) + dt*( ...
            - a3*gamma*wh(i) ...
            + sum_w );
    end

    %% ---- Save data ----
    X(:,k)  = x;
    V(:,k)  = v;
    X0(k)   = x0;
end
%% ========== Position tracking ==========
figure;
plot(t, X0, 'k', 'LineWidth', 2); hold on;
plot(t, X', '--', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Position');
title('Bipartite Consensus Tracking');
grid on;

%% ========== Velocity convergence ==========
figure;
plot(t, V', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Velocity');
title('Velocity Convergence');
grid on;
