%% This script simulates the CPS SR

clc; clear all; close all;

K = load('control.mat');
K = cell2mat(struct2cell(K));
ts = 0.002;
tt = 0:ts:40;  % 10 seconds to reach the initial point

tckp = 0.02;  % time needed for a checkpoint
trb = 0.016;  % time needed for a rollback

ttck = 0:ts:tckp;
ttrb = 0:ts:trb;

%% Kalman Filter setup

M=0.6;
g=9.81;
ff = M*g;


Ad = [1 0; ts 1]; Bd=[0 ts;(1/2)*(ts^2) 0]; Ao = zeros(2); Bo=zeros(2);
At = [Ad Ao Ao;Ao Ad Ao;Ao Ao Ad];

Bp = [Bd Bo Bo;...
      Bo Bd Bo;...
      Bo Bo Bd];

Bup = [0 0 0 0 0 ts]';  
Bua = zeros(6,1);

Cp = [0 1 0 0 0 0;...
      0 0 0 1 0 0;...
      0 0 0 0 0 1];
Ca = [1 0 0 0 0 0;...
      0 0 1 0 0 0;...
      0 0 0 0 1 0]; 

odeOpts = odeset(  ...
    'RelTol', 1e-6,  ...
    'AbsTol', 1e-7);

x_sp = [0 1 0 1 0 1 0 0 0 0 0 0]'; %starting setpoint
x0 = zeros(12,1);

% measurement noise
R=1e-5*eye(3);     %process noise variance
%v=sqrt(R)*randn(q,Ns); %process noise generation
Dv=3*1e-4;





% process noise
sigma_a2=1e-4;   %acceleration variance

Q=[ts^4/4 ts^3/2;ts^3/2 ts^2]*sqrt(sigma_a2); %process noise convariance matrix 
%Q=eye(2)*sqrt(sigma_a2);

Qt=[Q Ao Ao;Ao Q Ao;Ao Ao Q];

%Initial Condition and Initial Covariance matrix

P_p=0.001*eye(6); %initial covariance matrix 
hat_xp= zeros(6,1);...sqrt(P_p)*randn(6,1); % initial guess of the initial position
P_a=0.01*P_p;
hat_xa= zeros(6,1);...sqrt(P_a)*randn(6,1); % initial guess of the initial position

%% Simulation: Quadrotor takeoff, from zero
Ns=numel(tt);
uu=[];
xx_tot=x0';
hat_xx_tot=[hat_xp;hat_xa]';
for j=1:Ns
    yp = Cp*x0(1:6,1)+sqrt(R)*randn(3,1);  %position measurements
    ya = Ca*x0(7:12,1)+sqrt(R)*randn(3,1);  %angular vel. measurements
    
    [hat_xp,P_p] = rec_KF(hat_xp,P_p,yp,Qt,R,At,Bp,Cp,Dv,Bup,0); % position and velocity estimation
    [hat_xa,P_a] = rec_KF(hat_xa,P_a,ya,Qt,R,At,Bp,Ca,Dv,Bup,0); % orientation and angular velocity estimation
    
    hat_x0 = [hat_xp;hat_xa]; 
    % Compute the control law
    cu = -K*(hat_x0-x_sp);
    cu(1) = cu(1) + ff; 
    % Simulation of the real system
    x0 = x0 +  qds_dt(x0,cu)*ts;    %Euler
    % Save the control input, true state, and state estimation
    uu=[uu; cu'];
    xx_tot=[xx_tot;x0'];
    hat_xx_tot=[hat_xx_tot;hat_x0'];
end

%% Software Rejuvenation tracking Control Initialization 

x0 = xx_tot(end,:); %initial condition of the true system

hat_xp = hat_xx_tot(end,1:6)'; %initial condition of the state estimation
cu = uu(end,:)';
hat_xa = hat_xx_tot(end,7:12)';

hat_xp_ck = hat_xp;
hat_xa_ck = hat_xa;
hat_x0_ckp = [];
hat_0 = [hat_xp;hat_xa];
x0 = x0';
load('Pv.mat')

% waypoints
W = [0,1,0,1,0,1,0,0,0,0,0,0;...
     0,2,0,1,0,1,0,0,0,0,0,0;...
     0,2,0,2,0,1,0,0,0,0,0,0;...
     0,1,0,2,0,1,0,0,0,0,0,0;...
     0,1,0,1,0,1,0,0,0,0,0,0];
 
curr_waypoint = 1;

flag_A = 0;
flag_B = 0;

n_mode = 1; % the modes are: 1 Secure Control 
            %                2 Checkpoint 
            %                3 Mission Control
            %                4 Rollback

tt_sr=0;
tt_sr_v=0;

uu_sr=cu';
xx_sr_tot=[x0'];
hat_xx_sr_tot=[hat_x0'];
rb_counter = 1;
mc_counter = 1;
ck_counter = 1;
test_counter=1;
n_mode_tot=1;

P_a_ckp = [];
P_b_ckp = [];

x_sp_tot=x_sp';
x_sp_ckp = x_sp;
cu=[];
% The simulation of the mission starts here
while flag_A==0

    switch n_mode
        case 1
             yp = Cp*x0(1:6,1)+sqrt(R)*randn(3,1);  %position measurements
             ya = Ca*x0(7:12,1)+sqrt(R)*randn(3,1);  %angular vel. measurements
    
             [hat_xp,P_p] = rec_KF(hat_xp,P_p,yp,Qt,R,At,Bp,Cp,Dv,Bup,0); % position and velocity estimation
             [hat_xa,P_a] = rec_KF(hat_xa,P_a,ya,Qt,R,At,Bp,Ca,Dv,Bup,0); % orientation and angular velocity estimation
    
             hat_x0 = [hat_xp;hat_xa];
             if test_counter>=1500
           
                if (hat_x0-x_sp)'*P_value*(hat_x0-x_sp)<=0.005
                    n_mode = 3;
                    disp('here')
                    if (x_sp-W(curr_waypoint,:)')'*P_value*(x_sp-W(curr_waypoint,:)')<0.0001
                       curr_waypoint=curr_waypoint+1;
                    end
                    if curr_waypoint>size(W,1)
                       flag_A=1;
                    else
                        % -------------------------------------------------------------
                        % The next function is how to update the setpoints
                        % This function has to be substituted by the NN
                        % -------------------------------------------------------------
                        x_sp = update_xsp(x_sp,W(curr_waypoint,:)', P_value,0.01, 0.005)
                    end
                    ck_counter=1;
                    hat_x0_ckp = hat_x0;
                    P_p_ckp = P_p;
                    P_a_ckp = P_a;
                    x_sp_ckp = x_sp;
                    test_counter=1;
                end
             else
                 test_counter=test_counter+1;
             end
             cu = -K*(hat_x0-x_sp);
             cu(1) = cu(1) + ff;
             
        case 2
            if ck_counter<10
               ck_counter = ck_counter+1;
            else
                ck_counter = 1;
                n_mode = 3;
                mc_counter = 1;
            end
        case 3
            yp = Cp*x0(1:6,1)+sqrt(R)*randn(3,1);  %position measurements
            ya = Ca*x0(7:12,1)+sqrt(R)*randn(3,1);  %angular vel. measurements
    
            [hat_xp,P_p] = rec_KF(hat_xp,P_p,yp,Qt,R,At,Bp,Cp,Dv,Bup,0); % position and velocity estimation
            [hat_xa,P_a] = rec_KF(hat_xa,P_a,ya,Qt,R,At,Bp,Ca,Dv,Bup,0); % orientation and angular velocity estimation
            hat_x0 = [hat_xp;hat_xa];
            cu = -K*(hat_x0-x_sp);
            x_sp(2)
            cu(1) = cu(1) + ff;
            if mc_counter < 1000
                mc_counter = mc_counter+1;
            else
                mc_counter = 1;
                rb_counter = 1;
                n_mode = 4;
            end
        case 4
            if rb_counter < 5
               rb_counter = rb_counter+1;
            else
                rb_counter = 1;
                
                hat_xp = hat_x0_ckp(1:6,1);
                hat_xa = hat_x0_ckp(7:12,1);
                x_sp = x_sp_ckp;
                P_a = P_a_ckp;
                P_b = P_b_ckp;
                n_mode = 1;
            end
    end
                  
       % Dynamical system 
       x0 = x0 +  qds_dt(x0,cu)*ts;
       tt_sr = tt_sr + ts;
       tt_sr_v=[tt_sr_v, tt_sr];
       if tt_sr>40
           flag_A=1;
       end
       % Save the control input, true state, and state estimation
       uu_sr=[uu_sr;cu'];
       xx_sr_tot=[xx_sr_tot;x0'];
       hat_xx_sr_tot=[hat_xx_sr_tot;hat_x0'];
       n_mode_tot=[n_mode_tot, n_mode];
       x_sp_tot=[x_sp_tot;x_sp'];
end

figure
plot(uu_sr)


figure
plot(tt_sr_v,n_mode_tot)

figure
subplot(3,1,1)
plot(tt_sr_v,xx_sr_tot(1:end,2))
hold on
plot(tt_sr_v,hat_xx_sr_tot(1:end,2),'r')
plot(tt_sr_v,x_sp_tot(:,2))
plot(tt_sr_v,n_mode_tot/3)
xlabel('time - s')
ylabel('x')
subplot(3,1,2)
plot(tt_sr_v,xx_sr_tot(1:end,4))
hold on
plot(tt_sr_v,hat_xx_sr_tot(1:end,4),'r')
xlabel('time - s')
ylabel('y')
subplot(3,1,3)
plot(tt_sr_v,xx_sr_tot(1:end,6))
hold on
plot(tt_sr_v,hat_xx_sr_tot(1:end,6),'r')
xlabel('time - s')
ylabel('z')


figure
subplot(3,1,1)
plot(tt_sr_v,xx_sr_tot(1:end,1))
hold on
plot(tt_sr_v,hat_xx_sr_tot(1:end,1),'r')
xlabel('time - s')
ylabel('x')
subplot(3,1,2)
plot(tt_sr_v,xx_sr_tot(1:end,3))
hold on
plot(tt_sr_v,hat_xx_sr_tot(1:end,3),'r')
xlabel('time - s')
ylabel('y')
subplot(3,1,3)
plot(tt_sr_v,xx_sr_tot(1:end,5))
hold on
plot(tt_sr_v,hat_xx_sr_tot(1:end,5),'r')
xlabel('time - s')
ylabel('z')



figure
subplot(3,1,1)
plot(tt,xx_tot(1:end-1,2))
hold on
plot(tt,hat_xx_tot(1:end-1,2),'r')
xlabel('time - s')
ylabel('x')
subplot(3,1,2)
plot(tt,xx_tot(1:end-1,4))
hold on
plot(tt,hat_xx_tot(1:end-1,4),'r')
xlabel('time - s')
ylabel('y')
subplot(3,1,3)
plot(tt,xx_tot(1:end-1,6))
hold on
plot(tt,hat_xx_tot(1:end-1,6),'r')
xlabel('time - s')
ylabel('z')


figure
subplot(3,1,1)
plot(tt,xx_tot(1:end-1,1))
hold on
plot(tt,hat_xx_tot(1:end-1,1),'r')
xlabel('time - s')
title('Velocities')
ylabel('x')
subplot(3,1,2)
plot(tt,xx_tot(1:end-1,3))
hold on
plot(tt,hat_xx_tot(1:end-1,3),'r')
xlabel('time - s')
ylabel('y')
subplot(3,1,3)
plot(tt,xx_tot(1:end-1,5))
hold on
plot(tt,hat_xx_tot(1:end-1,5),'r')
xlabel('time - s')
ylabel('z')


figure
subplot(3,1,1)
plot(tt,xx_tot(1:end-1,8))
hold on
plot(tt,hat_xx_tot(1:end-1,8),'r')
xlabel('time - s')
ylabel('x')
subplot(3,1,2)
plot(tt,xx_tot(1:end-1,10))
hold on
plot(tt,hat_xx_tot(1:end-1,10),'r')
xlabel('time - s')
ylabel('y')
subplot(3,1,3)
plot(tt,xx_tot(1:end-1,12))
hold on
plot(tt,hat_xx_tot(1:end-1,12),'r')
xlabel('time - s')
ylabel('z')
%% Functions

function [dx] = qds_dt(x,U)

%% Body state vector

% u body frame velocity along x-axis
% v body frame velocity along y-axis
% w body frame velocity along z-axis

% p roll rate along x-axis body frame
% q pitch rate along y-axis body frame
% r yaw rate along z-axis body frame

%% Inertial state vector

% p_n inertial north position (x-axis)
% p_e inertial east position (y-axis)
% h   inertial altitude (- z-axis)

% phi   roll angle respect to vehicle-2
% theta pitch angle respect to vehicle-1
% psi   yaw angle respect to vehicle

%% Generic Quadrotor X PX4

M = 0.6;      % mass (Kg)
L = 0.2159/2; % arm length (m)

g  = 9.81;    % acceleration due to gravity m/s^2


m = 0.410;                 % Sphere Mass (Kg) 
R = 0.0503513;             % Radius Sphere (m)

m_prop = 0.00311;          % propeller mass (Kg)
m_m = 0.036 + m_prop;      % motor +  propeller mass (Kg)

Jx = (2*m*R)/5+2*L^2*m_m;
Jy = (2*m*R)/5+2*L^2*m_m;
Jz = (2*m*R)/5+4*L^2*m_m;


%% Rotation Matrices

phi=x(8); theta=x(10); psi=x(12);

Rp=[cos(theta)*cos(psi) sin(phi)*sin(theta)-cos(psi)*sin(psi) cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi);...
     cos(theta)*sin(psi) sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi) cos(phi)*sin(theta)*sin(psi)+sin(phi)*cos(psi);...
     sin(theta)  -sin(phi)*cos(theta) cos(phi)*cos(theta)];
 
Rv=[1 sin(phi)*tan(theta) cos(phi)*tan(theta);...
    0 cos(phi) -sin(phi);...
    0 sin(phi)*sec(theta) cos(phi)*sec(theta)];

%%

vect_p=Rv'*[x(7);x(9);x(11)];
p=vect_p(1); q=vect_p(2); r=vect_p(3);

vect_u=Rp'*[x(1);x(3);x(5)];
u=vect_u(1); v=vect_u(2); w=vect_u(3);

du=r*v-q*w-g*sin(theta);
dv=p*w-r*u+g*cos(theta)*sin(phi);
dw=q*u-p*v+g*cos(theta)*cos(phi)-(1/M)*U(1);

dp=((Jy-Jz)/Jx)*q*r+1/Jx*U(2);
dq=((Jz-Jx)/Jy)*p*r+1/Jy*U(3);
dr=((Jx-Jy)/Jz)*p*q+1/Jz*U(4);

vect_x=Rp'*[du;dv;dw];
vect_phi=Rv'*[dp;dq;dr];

dx=zeros(12,1);
dx(1)=vect_x(1);dx(3)=vect_x(2);dx(5)=vect_x(3);
dx(2)=x(1);dx(4)=x(3);dx(6)=x(5);
dx(7)=vect_phi(1); dx(9)=vect_phi(2); dx(11)=vect_phi(3);
dx(8)=x(7);dx(10)=x(9);dx(12)=x(11);

end


function [x_kk,P_kk] = rec_KF(x_k,P_k,y_k,Q,R,Ad,Bw,Cd,Dv,Bu,u)
% This function implements one step of the discrete time Kalman Filter 
% Inputs:
%        x_k : state estimation previous step
%        P_k : state estimation covariance matrix         
%        y_k : actual output measurements
%        Q,R : covariance matrices process and measurement noises
%   Ad,Bw,Cd : generic signal model
% Outputs:
%       x_kk : current state estimation
%       P_kk : current covariance matrix of the state estimation
    n=size(Ad,1);
    P=Ad*P_k*Ad'+Bw*Q*Bw';           %P(k+1|k)    
    Kk=P*Cd'*inv(Cd*P*Cd'+Dv*R*Dv');  %K(k+1)
    x_kk=Ad*x_k+Bu*u+Kk*(y_k-Cd*(Ad*x_k+Bu*u)); %x(k+1|k+1)
    P_kk=(eye(n)-Kk*Cd)*P;            %P(k+1|k+1) 
end

%% Update equilibrium points

function x_sp = update_xsp(x_sp,way_point, P,rho_m, rho_s)

   x_sp_old = x_sp;
   delta_wp= way_point-x_sp;

   
   norm_delta_wp = delta_wp'*P*delta_wp;
   
   alpha = (sqrt(rho_m)-sqrt(rho_s))/sqrt(norm_delta_wp);
   % setpoint update
   x_sp = x_sp+alpha*delta_wp;

   norm_delta_sp=(x_sp-x_sp_old)'*P*(x_sp-x_sp_old);
   if norm_delta_sp>norm_delta_wp
     x_sp = way_point;
   end
end