%%----------------------%%
% Input
%     R: image channel (current implementation using Red Channel)
%     ac: remove non-nuclei regions
% Output:
%    C_mask3: the foreground is nuclei which clump together or have
%    concave shape
%    cs4,rs4: the centers of isolated nuclei whith convex shape
%    A3: the isolated nuclei with convex shape
% Written by Hongming Xu
% ECE U of A
% feel free to use it
%%---------------------%%


function [C_mask3 cs4 rs4 A3]=XNucleiSeg_GL_Thresholding(R,ac,I,flagTMA)

% (i) global threshold using Otsu's method
% R1=imtophat(255-R,strel('disk',40));
% R=255-R1;
if ~exist('flagTMA','var')
    flagTMA=0;
end
%% HGMR module
R_hat=255-R;
% Opening by reconstruction
% S = [0 0 1 1 1 0 0;...
%     0 1 1 1 1 1 0;...
%     1 1 1 1 1 1 1;...
%     1 1 1 1 1 1 1;...
%     1 1 1 1 1 1 1;...
%     0 1 1 1 1 1 0;...
%     0 0 1 1 1 0 0];

S = [0 0 1 0 0;...
     0 1 1 1 0;...
     1 1 1 1 1 ;...
     0 1 1 1 0;...
     0 0 1 0 0];
Re=imerode(R_hat,S);
fRe=imreconstruct(Re,R_hat);%show(fRe);

% Closing-by-Reconstruction
fRerc=imcomplement(fRe);
fRerce=imerode(fRerc,S);
fRercbr=imcomplement(imreconstruct(fRerce,fRerc));

R=255-fRercbr;

TR=graythresh(R);%show(R)
RClogical=im2bw(R,TR);%show(RClogical)

%% get rid of white background
% bwTMA=R>230; %show(bwTMA);
% bwTMA=imfill(bwTMA,'holes');
% bwTMA=imopen(bwTMA,strel('disk',200));

%show(C_mask1);

if ~flagTMA
    C_mask1=imopen(~RClogical,strel('disk',2));
    C_mask1=bwareaopen(C_mask1,ac,8);
    C_mask1=imfill(C_mask1,'holes');%show(C_mask1);
else
    bwBK=I(:,:,1)>224;%show(bwBK); show(R);
    bwBK=bwBK&~RClogical;%show(~RClogical);
    bwBK=imerode(bwBK,strel('disk',2));
    C_mask1=~bwBK&~RClogical;
%show(C_mask1);
end
% C_mask1 = 255-R
% (ii) local threshold segmentation
[label2,n2]=bwlabel(C_mask1);%show(label2);
stats2=regionprops(label2,'BoundingBox');

% if flagTMA
%     stats2(1)=[];
% end
C_mask2=C_mask1;
%imshow(R)
for j=1:length(stats2)
    x=floor(stats2(j).BoundingBox(1));
    y=floor(stats2(j).BoundingBox(2));
    w=floor(stats2(j).BoundingBox(3));
    h=floor(stats2(j).BoundingBox(4));
    if x<1   %% make sure the bounding box is within the image
        x=1;
    end
    if y<1
        y=1;
    end
    tr=R(y:y+h,x:x+w);
%     fprintf('on the %dth obj\n',j);
   
    rr=im2bw(tr,graythresh(tr));
    C_mask2(y:y+h,x:x+w)=C_mask1(y:y+h,x:x+w)&(~rr);
%      hold on,
%      plot(stats2(j).BoundingBox(1),stats2(j).BoundingBox(2),'r*');
%      rectangle('Position',[stats2(j).BoundingBox(1),stats2(j).BoundingBox(2),stats2(j).BoundingBox(3),stats2(j).BoundingBox(4)],...
%         'EdgeColor','g','LineWidth',2);
end
% show(C_mask2);

C_mask2=imopen(C_mask2,strel('disk',2));
C_mask2=bwareaopen(C_mask2,round(ac*(2/3)),8);
C_mask2=imfill(C_mask2,'holes');


%(iii) find isolated nuclei centers with high fittness of convex shape
[label3,n3]=bwlabel(C_mask2);
stats3=regionprops(label3,'Solidity');
s3=cat(1,stats3.Solidity);
ind3=find(s3>0.95);
A3=ismember(label3,ind3);
c4=regionprops(A3,'centroid');
centroids4=cat(1,c4.Centroid);
cs4=centroids4(:,1);
rs4=centroids4(:,2);
C_mask3=C_mask2-A3;
end