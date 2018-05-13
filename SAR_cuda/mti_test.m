clear all
close all

Tp=1e-3;
fc=1275e6;
Vs=7.9e3;

C=3e8;

lambda=C/fc;
ant=Vs*Tp/2;

th=0;%2*ant/lambda*pi;
exp(1i*th)

fp_f=fopen('front.raw');
fp_r=fopen('rear.raw');
Nbin=1024;
Nhit=1024;


raw_data=fread(fp_f,Nbin*Nhit*2,'double');

i_data=raw_data(1:Nbin*Nhit);
q_data=raw_data(Nbin*Nhit+1:Nbin*Nhit*2);

i_data=reshape(i_data,[Nbin Nhit]);
q_data=reshape(q_data,[Nbin Nhit]);

raw_data=i_data+1i*q_data;

raw_data_f=raw_data.';

raw_data=fread(fp_r,Nbin*Nhit*2,'double');

i_data=raw_data(1:Nbin*Nhit);
q_data=raw_data(Nbin*Nhit+1:Nbin*Nhit*2);

i_data=reshape(i_data,[Nbin Nhit]);
q_data=reshape(q_data,[Nbin Nhit]);

raw_data=i_data+1i*q_data;

raw_data_r=raw_data.';

sub=abs(raw_data_r(2:end,:)*exp(-1i*th)-raw_data_f(1:end-1,:)*exp(1i*th));
add=abs(raw_data_r(2:end,:)*exp(-1i*th)+raw_data_f(1:end-1,:)*exp(1i*th));

figure;imagesc(sub)
figure;imagesc(add)

figure;imshow(add/9e9)

