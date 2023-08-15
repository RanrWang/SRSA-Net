% 
function [glog_mask]=np_glog_process(im,prior_mask,largeSigma,smallSigma,bandwidth)
image = im;
%R=255-I;
mask = prior_mask;
%mask = rgb2gray(mask);
if sum(sum(mask(:)))==0
    glog_mask = mask;
else
    R=image;
    %% gLoG seeds detection
    largeSigma=largeSigma;
    smallSigma=smallSigma;
    bandwidth=bandwidth;
    ns=XNucleiCenD_Clustering(R,mask,largeSigma,smallSigma,bandwidth);  %% To detect nuclei clumps
    %% marker-controlled watershed segmentation
    if isempty(ns)==1
        glog_mask = mask;%%if detection result is none, save the prior result
    else
        ind=sub2ind(size(R),ns(1,:),ns(2,:));
        bs4=zeros(size(R));
        bs4(ind)=1;
        [bnf,blm]=XWaterShed(mask,bs4);
        %% show segmentations
        % bb=bwperim(Nmask);  % for debugging finial segmentations
        mask(blm)=0;
        glog_mask = mask;
    end
end

end

