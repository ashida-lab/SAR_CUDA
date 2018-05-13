clear all
close all

fp=fopen('test.raw');
Nbin=1024;
Nhit=1024;


raw_data=fread(fp,Nbin*Nhit*2,'double');

i_data=raw_data(1:Nbin*Nhit);
q_data=raw_data(Nbin*Nhit+1:Nbin*Nhit*2);

i_data=reshape(i_data,[Nbin Nhit]);
q_data=reshape(q_data,[Nbin Nhit]);

raw_data=i_data+1i*q_data;

raw_data=raw_data.';

figure;imagesc(abs(raw_data))

a_fft=fft2(raw_data);

figure;imagesc(abs(a_fft));

a1=ifft2(a_fft(1:Nhit/2,:));
a2=ifft2(a_fft(Nhit/2+1:Nhit,:));

figure;imagesc(abs(a1))
figure;imagesc(abs(a2))

figure;imagesc(real(a1)-real(a2))

figure;imagesc(imag(a1)+imag(a2))

figure;imagesc(angle(a1-a2))