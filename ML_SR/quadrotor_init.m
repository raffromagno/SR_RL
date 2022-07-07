function [quadrotor_full_sys,quadrotor_att_sys, Acl,Aacl,K,Ka] = quadrotor_init(params)
%quad_rotor_init: Provides the constrained
%   Detailed explanation goes here

m = params.m;...0.6;      % mass (Kg)
L = params.L;
g = params.g;
Jx = params.Jx;
Jy = params.Jy;
Jz = params.Jz;

%L = 0.2159/2; % arm length (m)

%g  = 9.81;    % acceleration due to gravity m/s^2


%M = 0.410;                 % Sphere Mass (Kg) 
%R = 0.0503513;             % Radius Sphere (m)

%m_prop = 0.00311;          % propeller mass (Kg)
%m_m = 0.036 + m_prop;      % motor +  propeller mass (Kg)

%Jx = (2*M*R)/5+2*L^2*m_m;
%Jy = (2*M*R)/5+2*L^2*m_m;
%Jz = (2*M*R)/5+4*L^2*m_m;

%% Linearized Model in Hoovering mode
n1=6;
A_p=[0 0 0 0 0 0;...
     1 0 0 0 0 0;...
     0 0 0 0 0 0;...
     0 0 1 0 0 0;...
     0 0 0 0 0 0;...
     0 0 0 0 1 0];
 
 Aa=A_p;
 
 A_i=[0 0 0 -g 0 0;...
      0 0 0 0 0 0;...
      0 g 0 0 0 0;...
      0 0 0 0 0 0;...
      0 0 0 0 0 0;...
      0 0 0 0 0 0];
 
 Bp=[0 0 0 0 -1/m 0 0 0 0 0 0 0]';
 Ba=[0 0 0 0 0   0  1/Jx 0 0 0 0 0;...
     0 0 0 0 0   0  0  0  1/Jy 0 0 0;...
     0 0 0 0 0   0  0  0   0  0  1/Jz 0]';
 
 A=[A_p A_i; zeros(n1) Aa];
 B=[Bp Ba];
 C=[0 1 0 0 0 0 0 0 0 0 0 0;...
    0 0 0 1 0 0 0 0 0 0 0 0;...
    0 0 0 0 0 1 0 0 0 0 0 0;...
    0 0 0 0 0 0 0 0 0 0 0 1];


n=size(A,1);
p=size(B,2);
q=size(C,1);
D=zeros(q,p);

%% LQR Controller


quadrotor_full_sys = ss(A,B,C,D);
quadrotor_att_sys = ss(Aa,Ba(7:12,:),eye(n1),zeros(n1,size(Ba,2)));

Qq = 1;
Rr = 100;
Q = Qq*eye(n);
R = Rr*eye(p);
[K,~,~] = lqr(quadrotor_full_sys,Q,R);
Q1 = Qq*eye(n1);
R1 = Rr*eye(size(Ba,2));
[Ka,~,~] = lqr(quadrotor_att_sys,Q1,R1);

% Closed-loop

Acl=A-B*K;        % full system

Aacl=Aa-Ba(7:12,:)*Ka;    % attitude control

end

