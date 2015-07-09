% GRASP (Golden-angle RAdial Sparse Parallel MRI)
% Combination of compressed sensing, parallel imaging and golden-angle
% radial sampling for fast dynamic MRI.
% This demo will reconstruct one slice of a contrast-enhanced liver scan.
% Radial k-space data are continously acquired using the golden-angle
% scheme and retrospectively sorted into a time-series of images using
% a number of consecutive spokes to form each frame. In this example, 21 
% consecutive spokes are used for each frame, which provides 28 temporal
% frames. The reconstructed image matrix for each frame is 384x384. The 
% undersampling factor is 384/21=18.2. TheNUFFT toolbox from Jeff Fessler 
% is employed to reconstruct radial data
% 
% Li Feng, Ricardo Otazo, NYU, 2012

clear all;close all;
addpath('nufft_toolbox/');
% define number of spokes to be used per frame (Fibonacci number)
nspokes=21;
% load radial data
load liver_data.mat

kdata_fileID = fopen('kdata_dump_matlab', 'wt');
fprintf(kdata_fileID, '%f + %fi\n', real(kdata), imag(kdata));
fclose(kdata_fileID)
b1_fileID = fopen('b1_dump_matlab', 'wt');
fprintf(b1_fileID, '%f + %fi\n', real(b1), imag(b1));
fclose(b1_fileID)
%k_fileID = fopen('k_dump_matlab', 'wt');
%fprintf(k_fileID, '%f + %fi\n', real(k), imag(k));
%fclose(k_fileID)
w_fileID = fopen('w_dump_matlab', 'wt');
fprintf(w_fileID, '%f\n', w);
fclose(w_fileID)
exit(0)

b1=b1/max(abs(b1(:)));
% data dimensions
[nx,ntviews,nc]=size(kdata);
% density compensation
for ch=1:nc,kdata(:,:,ch)=kdata(:,:,ch).*sqrt(w);end
% number of frames
nt=floor(ntviews/nspokes);
% crop the data according to the number of spokes per frame
kdata=kdata(:,1:nt*nspokes,:);
k=k(:,1:nt*nspokes);
w=w(:,1:nt*nspokes);
% sort the data into a time-series
for ii=1:nt
    kdatau(:,:,:,ii)=kdata(:,(ii-1)*nspokes+1:ii*nspokes,:);
    ku(:,:,ii)=k(:,(ii-1)*nspokes+1:ii*nspokes);
    wu(:,:,ii)=w(:,(ii-1)*nspokes+1:ii*nspokes);
end
kdatau_fileID = fopen('kdatau_dump', 'w');
fprintf(kdatau_fileID, '%f\n', kdatau);
exit(0)
% multicoil NUFFT operator
param.E=MCNUFFT(ku,wu,b1);
% undersampled data
param.y=kdatau;
clear kdata kdatau k ku wu w
% nufft recon
recon_nufft=param.E'*param.y;
% parameters for reconstruction
param.W = TV_Temp();param.lambda=0.25*max(abs(recon_nufft(:)));
% number of iterations
param.nite = 8;
param.display=1;
fprintf('\n GRASP reconstruction \n')
tic
recon_cs=recon_nufft;
for n=1:3,
	recon_cs = CSL1NlCg(recon_cs,param);
end
toc
recon_nufft=flipdim(recon_nufft,1);
recon_cs=flipdim(recon_cs,1);

% display 4 frames
recon_nufft2=recon_nufft(:,:,1);recon_nufft2=cat(2,recon_nufft2,recon_nufft(:,:,7));recon_nufft2=cat(2,recon_nufft2,recon_nufft(:,:,13));recon_nufft2=cat(2,recon_nufft2,recon_nufft(:,:,23));
recon_cs2=recon_cs(:,:,1);recon_cs2=cat(2,recon_cs2,recon_cs(:,:,7));recon_cs2=cat(2,recon_cs2,recon_cs(:,:,13));recon_cs2=cat(2,recon_cs2,recon_cs(:,:,23));
figure;
subplot(2,1,1),imshow(abs(recon_nufft2),[]);title('Zero-filled FFT')
subplot(2,1,2),imshow(abs(recon_cs2),[]);title('GRASP')
