
clear all
clc
addpath(genpath('./ColorFeatures'));

imName = '0_18_18957.jpg';
lab_im = RGB2Lab(imread(imName));%%LAB image
rgb_im = double(imread(imName));%%RGB image
pb_im = im2double(imread([imName(1:end-4) '_pb.png']));%% pb edge image after bi-segmenting by otsu
sulabel_im = ReadDAT(size(pb_im),[imName(1:end - 4) '.dat']);%Superpixel Label
%% Obtain interesting points
thresh = 26; % for elimate the side point
corner_im2 = getsalientpoints(rgb_im);
corner_im = elimatepoint(corner_im2,thresh); % elimate the points closing to the boundary of images
%% Calculate prior map
[row,col] = size(corner_im);
[y,x] = ind2sub([row,col],find(corner_im == 1));
dt = DelaunayTri(x,y);
if(~size(dt,1))
    return;
end
[k,av] = convexHull(dt);
BW = roipoly(corner_im,x(k),y(k));
pixel = regionprops(BW,'all');
ind = pixel.PixelIdxList;
out_ind = setdiff(1:row*col,ind);
sal_super = priormap(lab_im, ind, sulabel_im,pb_im,rgb_im);

%% Revised observation likelihood
convexhull = ReviseConvexHull(sal_super, rgb_im, sulabel_im);
hull_pixel = regionprops(convexhull,'all');
ind = [];
out_ind = [];
for i = 1:length(hull_pixel)
   ind = [ind;hull_pixel(i).PixelIdxList];%%找到mask的白色区域
end
out_ind = setdiff(1:row*col,ind);
[PrI_sal, PrI_bk,PrO_sal,PrO_bk] = likelihoodprob(rgb_im, ind,out_ind);

%% Bayesian combination
psal_I = sal_super(ind);
psal_O = sal_super(out_ind');
Pr_0=(PrI_sal.*psal_I)./(PrI_sal.*psal_I+PrI_bk.*(1 - psal_I));%so called saliency 窗内的saliency
Pr_B=(PrO_sal.*psal_O)./(PrO_sal.*psal_O+PrO_bk.*(1-psal_O));%so called saliency 窗外的saliency
saliencymap = zeros(row,col);
saliencymap(ind) = Pr_0;
saliencymap(out_ind) = Pr_B;
saliencymap = (saliencymap - min(saliencymap(:)))/(max(saliencymap(:)) - min(saliencymap(:)));
figure
imshow(saliencymap);
