% define
showRotated=2; % 1 for oself rotated, 2 for bestROI model, 3 for full brain
zz = 1;
ideal_rotation = 86.59; % ideal rotation in degrees 
ideal_rotation_rr = deg2rad(ideal_rotation);
subjid = sprintf('subj%02d',zz);   % which subject
cmap   = jet(256);   % colormap for ROIs
rng    = [0 19];      % should be [0 N] where N is the max ROI index
rng_reduced = [0 12];
roilabels = {'V1v' 'V1d' 'V2v' 'V2d' 'V3v' 'V3d' 'hV4' 'VO-1' 'VO-2' 'PHC-1' 'PCH-2' 'LO-1' 'LO-2' 'TO-1' 'TO-2' 'PHC-3' 'PHC-4' 'TO-3' 'PHC-5'};  % 1 x N cell vector of strings
roilabels_reduced = {'V1' 'V2' 'V3' 'hV4' 'VO-1' 'VO-2' 'PHC-1' 'PCH-2' 'LO-1' 'LO-2' 'TO-1' 'TO-2'};
prfang = {mod(180-cvnloadmgz(sprintf('%s/%s/label/lh.prfangle.mgz',cvnpath('freesurfer'),subjid)),360) ...  % one colormap for both hemis
          cvnloadmgz(sprintf('%s/%s/label/rh.prfangle.mgz',cvnpath('freesurfer'),subjid))};

if showRotated==3   % Need to fix this: create a new full mask with some value to
    % manage the deleted voxel, and fix the train_test_mask
    mgznames = {
        %   prfang
        %   'prfeccentricity 
        prfang
        'prfeccentricity'
        'prfR2'
      %  'oself.fit_with.rotated'
        % 'oself.fit_with.rotated'
        %  'oself.intercept'
        %  'oself.slope'
  %      'mds_ang.fullTO-1'

        %  'oself.mds_ang.90'
 %       'mds_ecc.fullTO-1'
        %   'oself.sigma'
        'x0.fullTO-1' 
        
        'y0.fullTO-1'
        'test_var_explained.fullTO-1'
        
        %   'oself.var_explained.rotated'
        };           % quantities of interest (1 x Q)
end

if showRotated==2
    mgznames = {
        %   prfang
        %   'prfeccentricity 
        prfang
        'prfeccentricity'
        'prfR2'
      %  'best_roi.roi.rotated'
        % 'oself.fit_with.rotated'
        %  'oself.intercept'
        %  'oself.slope'
     %   'best_roi.mds_ang.rotated'


    %    'best_roi.mds_ecc.rotated'
     %   'best_roi.sigma.rotated'
        'best_roi.x0.rotated' 
        
        'best_roi.y0.rotated'
        'best_roi.test_var_explained.rotated'
   %     'best_roi.noise_ceilling.rotated'
        'best_roi.voxel_performance.rotated'
   %     'best_roi.sigma.rotated'
    
        };           % quantities of interest (1 x Q)
end
if showRotated==1
    mgznames = {
        %   prfang
        %   'prfeccentricity 
        prfang
        'prfeccentricity'
        'prfR2'
   %     'oself.fit_with.rotated'
        % 'oself.fit_with.rotated'
        %  'oself.intercept'
        %  'oself.slope'
     %   'oself.mds_ang.rotated'

        %  'oself.mds_ang.90'
   %     'oself.mds_ecc.rotated'
        
        'oself.x0.rotated' 
        'oself.y0.rotated'
        'oself.test_var_explained.rotated'
     %   'oself.noise_ceilling.rotated'
        'oself.voxel_performance.rotated'
        'oself.sigma.rotated'
        
        %   'oself.var_explained.rotated'
        };           % quantities of interest (1 x Q)
end 
if showRotated==0
    mgznames = {
        %   prfang
        %   'prfeccentricity'
        prfang
        'prfeccentricity'
        'prfR2'

        'oself.fit_with.notrotated'
        
        %  'oself.intercept'
        %  'oself.slope'
        % 'oself.mds_ang.notrotated'
        'oself.mds_ang.notrotated'

        %  'oself.mds_ang.90'
        % 'oself.mds_ecc.notrotated'
        'oself.mds_ecc.notrotated'
        %   'oself.sigma'
        'oself.x0.notrotated'
        'oself.y0.notrotated'
        'oself.test_var_explained.notrotated'
       
        %   'oself.var_explained'
        };           % quantities of interest (1 x Q)
end
mgznames = mgznames';
views_to_trim = [];
ang_view = 1;
ecc_view = 2;
r2_view = 3;

vrngs = {};
crngs = {};
cmaps = {};
threshs = {};
otherthreshs={};
overlays = {};



%TRY DOING THIS TO SIMPLIFY RELATIONSHIPS BETWEEN PARAMS
%Parameter 1: Spatial Polar angle. Key = 3 ? 1 is ROIs and 2 is nothing
paramNum=1;
vrngs{paramNum}= 360;
crngs{paramNum}=[0 360];
cmaps{paramNum}=cmfanglecmapRH;
threshs{paramNum}= vrngs{paramNum} * 0;
otherthreshs{paramNum}= [5, 3];
overlays{paramNum} = 1;


% Parameter 2 : Spatial Eccentricty. Key = 4 
paramNum=2;
vrngs{paramNum}= 1000;
crngs{paramNum}=[0 12];
cmaps{paramNum}=fliplr(jet(256));
threshs{paramNum}= vrngs{paramNum} * 0; % 12 is apparently the magnitude used in literature even if the data is capped at 1000? When using 1k it dissolves
otherthreshs{paramNum}= [5, 3];
overlays{paramNum} = 1;

% Parameter 3 : Goodnest of fit of pRF; thresholded. Key = 5
paramNum=3;
vrngs{paramNum}= 100;
crngs{paramNum}=[0 100];
cmaps{paramNum}=hot(256);
threshs{paramNum}= vrngs{paramNum} * 0.05;
overlays{paramNum} = 1;
 
% % Not a param, just shows ROI reduced (so 12 ROIs instead of 15); Key = 6
% paramNum=4;
% vrngs{paramNum}= 12;
% crngs{paramNum}=[0 12];
% cmaps{paramNum}=jet(256);
% threshs{paramNum}= vrngs{paramNum} * 0;
% overlays{paramNum} = 1;

% Parameter 5 : Polar angle of Voxel (mds). Key = 7 (6 is custom ROIS, named ROIS_Reduced)
% paramNum=4;
% vrngs{paramNum}= 360;
% crngs{paramNum}=[0 360];
% cmaps{paramNum}=cmfanglecmapRH;
% threshs{paramNum}= vrngs{paramNum} * 0.05;
% %otherthreshs{paramNum}= [0, 8];
% overlays{paramNum} = 1;

% paramNum=6;
% vrngs{paramNum}= 360;
% crngs{paramNum}=[0 360];
% cmaps{paramNum}=cmfanglecmapRH;
% threshs{paramNum}= vrngs{paramNum} * 0.05;
% otherthreshs{paramNum}= [0, 8];
% overlays{paramNum} = 1;


% Parameter 6 :  Voxel eccentricity (mds). Key = 8
% paramNum=5;
% vrngs{paramNum}= 2;
% crngs{paramNum}=[0 2];
% cmaps{paramNum}=fliplr(jet(256)); %
% threshs{paramNum}= vrngs{paramNum} * 0.00;
% %otherthreshs{paramNum}= [0, 8]; Removed it for  visualization
% overlays{paramNum} = 1;

%  paramNum=8;
%  vrngs{paramNum}= 2;
%  crngs{paramNum}=[0 2];
%  cmaps{paramNum}=fliplr(jet(256));
%  threshs{paramNum}= vrngs{paramNum} * 0.10;
%  otherthreshs{paramNum}= [0, 8];
%  overlays{paramNum} = 1;

%Parameter 7 : x0. Key = 9 
paramNum=4;
vrngs{paramNum}= 2;
crngs{paramNum}=[0 2];
cmaps{paramNum}=hsv(256);
threshs{paramNum}= vrngs{paramNum} * 0;
%otherthreshs{paramNum}= [0, 8];
overlays{paramNum} = 1;

% % Parameter 8 : y0. Key = 0
paramNum=5;
vrngs{paramNum}= 2;
crngs{paramNum}=[0 2];
cmaps{paramNum}=hsv(256);
threshs{paramNum}= vrngs{paramNum} * 0;
overlays{paramNum} = 1;

% Parameter 9 : test variance explained. Key = 0
paramNum=6;
vrngs{paramNum}= 2;  % check if this allo for negative values -> it does
crngs{paramNum}=[0.5 1.5]; % This is centered on 1, so [1-threshold 1+threshold]; but 0s are the masked values -> thing change if I put negative values
cmaps{paramNum}=coolhotCmap(0,256);
threshs{paramNum}= vrngs{paramNum} *0.25; % are they negative values before/after thresholding
overlays{paramNum} = 1;



% Parameter 10: noise ceilling 
% paramNum=7;
% vrngs{paramNum}= 2;  
% crngs{paramNum}=[0 0.5]; 
% cmaps{paramNum}=jet( 256);
% threshs{paramNum}= vrngs{paramNum} *0; 
% overlays{paramNum} = 1;


% Parameter 11: voxel peformance (variance explained / noise ceilling)
paramNum=7;
vrngs{paramNum}= 2;  
crngs{paramNum}=[0 2]; 
cmaps{paramNum}=coolhotCmap(0,256);
threshs{paramNum}= vrngs{paramNum} *0; 
overlays{paramNum} = 1;

% Parameter 12: sigma (size of response function)
paramNum=8;
vrngs{paramNum}= 12;  
crngs{paramNum}=[0 12]; 
cmaps{paramNum}=jet(256);
threshs{paramNum}= vrngs{paramNum} *0; 
overlays{paramNum} = 1;


vrngs
% 
% vrngs = {  
% %     360 
% %     1000 
%     360 
%     1000 
%     100 
%     12   % rois_reduced
% %     500   % intercept
% %     500   % slope
%     360  % mds_ang
% %     360  % mds_ang
%     2    % mds_ecc
% %     6  % sigma
%     2
%     2
% %     1    % test_var_explained
% %     1    % var_explained
% };
% vrngs
% vrngs = vrngs';
r2_threshold = 0.05 * vrngs{r2_view};
% crngs = { 
% %   [0 360] 
% %   [0 12] 
%   [0 360] %1: Spatial polar angle
%   [0 12]  %2: Spatial eccentricity
%   [0 50] 
%   [0 12]     % rois_reduced
% %   [0 500]   % intercept
% %   [0 500]   % slope
% %   [0 360]    % mds_ang
%   [0 360]    % mds_ang
%   [0 2]      % mds_ecc
% %   [0 6]    % sigma
%   [0 2]
%   [0 2]
% %   [0 1]      % test_var_explained
% %   [0 1]      % var_explained
% 
% };          % ranges for the quantities (1 x Q)
%crngs = crngs';
% cmaps = { 
% %   cmfanglecmapRH 
% %   fliplr(jet(256)) 
%   cmfanglecmapRH 
%   fliplr(jet(256)) 
%   hot(256) 
%   jet(256)   % rois_reduced
% %   fliplr(hot(256))  % intercept
% %   hot(256)  % slope
% %   cmfanglecmapRH  % mds_ang
%   cmfanglecmapRH  % mds_ang
%   fliplr(jet(256))  % mds_ecc
% %   hot(256)  % sigma
%   hsv(256)
%   hsv(256)
% %   hot(256)  % test_var_explain    ed
% %   hot(256)  % var_explained
% };   % colormaps for the quantities (1 x Q)
%cmaps = cmaps';
% threshs = { 
%   vrngs{1}* 0 
%   vrngs{2}* 0   % 12 is apparently the magnitude used in literature even if the data is capped at 1000? When using 1k it dissolves
%   vrngs{3}* 0.05
%   vrngs{4}* 0   % 12 is apparently the magnitude used in literature even if the data is capped at 1000? When using 1k it dissolves
%   vrngs{5}* 0 
%   vrngs{6}* 0 
%   vrngs{7}* 0 
%   vrngs{8}* 0 
% %   vrngs{9}* 0
% %   [] [] [] [] [] [] [] []
% };   % thresholds
%threshs = threshs';
% overlays = { 
%     1 
%     1 
%     1  
%     1 
%     1 
%     1 
%     1 
%     1 
%     1 
% }; % overlays
% overlays = overlays';
roivals = [];                                                                    % start with a blank slate?
% roivals =
% cvnloadmgz(sprintf('%s\\%s\\label\\customrois\\*.%s.testrois.mgz',cvnpath('freesurfer'),subjid,subjid));
% % load in an existing file? Windows
roivals = cvnloadmgz(sprintf('%s/%s/label/customrois/*.%s.testrois.mgz',cvnpath('freesurfer'),subjid,subjid));  % load in an existing file? Linux

% do it


curlegend = figureprep([0 0 300 600],1); hold on;
imagesc((1:rng_reduced(2))',rng_reduced);
set(gca,'YTick',1:rng_reduced(2),'YTickLabel',roilabels_reduced);
set(gca,'XTick',[]);
colormap(cmap);
axis image tight;
set(gca,'YDir','normal');
title('rois_reduced colormap');

cvndefinerois;