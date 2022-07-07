%% This script computes the controller K and the Matrix P for the CPS LQR 
%% problem

clc;
clear all;

load('jmavsim_quadrotor_params.mat');

[quadrotor_full_sys,quadrotor_att_sys, Acl,Aacl,K,Ka] = quadrotor_init(params);

save('control.mat','K')
