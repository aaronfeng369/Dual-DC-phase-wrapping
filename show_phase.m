clc
clear
close all
% Show the results of Dual-DC
% Jun-25-2023===Shengyuan Ma===Original Code
%%

load U_p0.mat
figure,
subplot(2,2,1)
imagesc(squeeze(angle(U(:,:,2,1,1,3))));axis off;axis image;title('Raw phase (no noise)')
load W_p0.mat
subplot(2,2,2)
imagesc(squeeze(real(W(:,:,2,1,3))));axis off;axis image;title('After unwrapping (no noise)')

load U_p0.4.mat
subplot(2,2,3)
imagesc(squeeze(angle(U(:,:,2,1,1,3))));axis off;axis image;title('Raw phase (with noise)')
load W_p0.4.mat
subplot(2,2,4)
imagesc(squeeze(real(W(:,:,2,1,3))));axis off;axis image;title('After unwrapping (with noise)')