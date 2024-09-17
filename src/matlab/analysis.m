subjid = 'subj01';
sigm = {cvnloadmgz(sprintf('%s/%s/label/lh.best_roi.sigma.rotated.mgz',cvnpath('freesurfer'),subjid)) ...  % one colormap for both hemis
          cvnloadmgz(sprintf('%s/%s/label/rh.best_roi.sigma.rotated.mgz',cvnpath('freesurfer'),subjid))};

mapNames_extended = ['V1' 'V2d' 'V2v' 'V3d' 'V3v' "hV4" "VO-1"    "VO-2"    "PHC-1"    "PHC-2"    "LO-1"    "LO-2" "TO-1"    "TO-2"];
new_order = {'V1' 'V2d' 'V2v' 'V3d' 'V3v' 'hV4' 'VO_1'    'VO_2'    'PHC_1'   'PHC_2'    'LO_1'   'LO_2' 'TO_1'    'TO_2'};
spearman_lh_oself = readtable('lh_spearman_rho_oself.csv');
spearman_lh_oself(:, 1) = [];
spearman_lh_oself = spearman_lh_oself(:, new_order);
lhspearmanrhooself = table2array(spearman_lh_oself);

spearman_rh_oself = readtable('rh_spearman_rho_oself.csv');
spearman_rh_oself(:, 1) = [];
spearman_rh_oself = spearman_rh_oself(:, new_order);
rhspearmanrhooself = table2array(spearman_rh_oself);

spearman_lh = readtable('lh_spearman_rho.csv');
spearman_lh(:, 1) = [];
spearman_lh = spearman_lh(:, new_order);
lhspearmanrho = table2array(spearman_lh);

spearman_rh = readtable('rh_spearman_rho.csv');
spearman_rh(:, 1) = [];
spearman_rh = spearman_rh(:, new_order);
rhspearmanrho = table2array(spearman_rh);



% Roi colors 
col = cell(12,1);
col{1} = [0 0 0.843137];
col{2} = [0 0.168627 1];
col{3} = [0 0.513725 1];
col{4} = [0 0.843137 1];
col{5} = [0.168627 1 0.827541];
col{6} = [0.513725 1 0.42353];
col{7} = [0.843137 1 0.152941];
col{8} = [1 0.827451 0];
col{9} = [1 0.482353 0];
col{10} = [1 0.152941 0];
col{11} = [0.827451 0 0];
col{12} = [0.498039 0 0];

col_ex = cell(14,1);
col_ex{1} = [0 0 0.843137];
col_ex{2} = [0 0.168627 1];
col_ex{3} = [0 0.168628 1];
col_ex{4} = [0 0.513725 1];
col_ex{5} = [0 0.513726 1];
col_ex{6} = [0 0.843137 1];
col_ex{7} = [0.168627 1 0.827541];
col_ex{8} = [0.513725 1 0.42353];
col_ex{9} = [0.843137 1 0.152941];
col_ex{10} = [1 0.827451 0];
col_ex{11} = [1 0.482353 0];
col_ex{12} = [1 0.152941 0];
col_ex{13} = [0.827451 0 0];
col_ex{14} = [0.498039 0 0];



R2_prf = cat(3, R2lh, R2rh);
sigma_prf = cat(3, prfsizelh, prfsizerh);
%Add to these comparisons the fits (VE) and pRF sizes (sigma) of the visual field mapping pRF model
testVE_oself_rotated =cat(3, testvariancerotatedlh, testvariancerotatedrh); % Variance explain oself model
testVE_oself_notrotated = cat(3, testvariancenotrotatedlh, testvariancenotrotatedrh);
testVE_bestroi = cat(3, testvariancebestroilh, testvariancebestroirh);
sigma_oself_notrotated = cat(3, sigmanotrotatedlh, sigmanotrotatedrh);
sigma_oself_rotated = cat(3, sigmarotatedlh, sigmarotatedrh); % sigma (size of rec field) oself
sigma_bestroi = cat(3, sigmabestroilh, sigmabestroirh);
noiseceiling_bestroi = cat(3, noiseceilingbestroilh, noiseceilingbestroirh);
noiseceiling_oself = cat(3, noiseceilingoselflh, noiseceilingoselfrh);
VP_best_roi = cat(3, voxelperformancebestroilh, voxelperformancebestroirh);
VP_oself_rotated = cat(3, voxelperformanceoselflh, voxelperformanceoselfrh);
spearmanrho = cat(3, lhspearmanrho, rhspearmanrho); % rho value of the correlation between mds and cort surf
spearmanrho_oself = cat(3, lhspearmanrhooself, rhspearmanrhooself);

%So we can use the same code below for any parameter
%data=testVE;
mean_NC = squeeze(mean(noiseceiling_bestroi, [1,3]));  % gets means NC for each ROI
data = spearmanrho;


%Setting up ANOVA structures
%23/05: modified it a bit to allow different ROI map size 
hemisphereGroups=cat(3, ones(size(data(:,:,1))), ones(size(data(:,:,1)))*2);
tmp=1:size(data, 1);
subjectLabels=cat(3, repmat(tmp(:), [1,size(data, 2)]), repmat(tmp(:), [1,size(data, 2)]));
if size(data, 2) == 14
    mapLabels=cat(3, repmat(1:length(mapNames_extended), [length(tmp),1]), repmat(1:length(mapNames_extended), [length(tmp),1]));
else
    mapLabels=cat(3, repmat(1:length(mapNames), [length(tmp),1]), repmat(1:length(mapNames), [length(tmp),1]));
end
mapLabels=mapLabels(:);  % need to add the extra ROI for spearman 
subjectLabels=subjectLabels(:);
hemisphereLabels=hemisphereGroups(:); 
%subjectLabels=subjectOrder(subjectLabels);
if size(data, 2) == 14
    mapLabelsText=mapNames_extended(mapLabels);
else
    mapLabelsText=mapNames(mapLabels);
end 
hemiTmp={'L', 'R'};
hemisphereLabels=hemiTmp(hemisphereLabels);


%Now input to mulit-factor anova
%General code for 3 factors, use when hemisphere difference is significant
[p, tmp, statsOut] = anovan(data(:),{hemisphereLabels subjectLabels mapLabelsText}, 'varnames', {'hemisphere', 'subject', 'map'});
%Multiple comparison tests on ANOVA output
figure; results=multcompare(statsOut, 'Dimension', [1 3])
if data == sigma  % works, allow for different axis
    axis([0 8 1 24.5])
else
    axis([-.07 .07 1 24.5])
end
axis square

%General code for 2 factors, use when hemisphere difference is NOT significant
[p, tmp, statsOut] = anovan(data(:),{subjectLabels mapLabelsText}, 'varnames', {'subject', 'map'});
%Multiple comparison tests on ANOVA output
figure; results=multcompare(statsOut, 'Dimension', [2])
% axis([0 pi/2 0.5 18.5])
if all(data(:)==sigma(:))
    axis([0 8 0.5 12.5])
elseif all(data(:)==testVE(:)) | all(data(:)==spearmanrho(:)) 
    axis([-.07 .07 0.5 12.5])
end
axis square

%Easy access to significance of differences to a particular map
thisMap=10;
includesThisMap=results(:,1)==thisMap | results(:,2)==thisMap;
results(includesThisMap,6)


%%% Make a plot 
[p, tmp, statsOut] = anovan(data(:),{subjectLabels mapLabelsText}, 'varnames', {'subject', 'map'});
figure;
[results] =multcompare(statsOut, 'Dimension', [2]);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:2*length(mapNames)-1; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end
means = fliplr(means);
CIs = flipud(CIs);

figure; plot(means, 'ok', 'MarkerFaceColor', 'k'); %plot(mean_NC,'_', 'MarkerSize', 10, 'LineWidth', 4, 'MarkerFaceColor' ,'k');
hold on;
for n = 1:length(mapNames)
    h = line([n n], [CIs(n,1) CIs(n,2)]);
    get(h)
    h.Color = col{n};
    h.LineWidth = 2;
    set(h)
end
title(['Within ROI Model']);
%axis([0.5 length(mapNames)+0.5 0 8]);
xlabel(['Visual Field Maps'])
ylabel(['Model Performance'])
axis([0.5 length(mapNames)+0.5 0 0.6]);
axis square
xticks([1:length(mapNames)]);
xticklabels(mapNames);
xtickangle(45);


%A SEPARATE SET OF TESTS for whether the fits are significantly different
%from zero
for n=1:max(mapLabels)
    mapValues=data(mapLabels==n);
    ps(n)=signrank(mapValues);
    medians(n)=median(mapValues);
end
[~, ~, ~, ps_fdr]=fdr_bh(ps);
ps_fdr(medians<0)=-ps_fdr(medians<0);
ps(medians<0)=-ps(medians<0);
[ps(:) ps_fdr(:)]







%%% Make a plot  for spearman
[p, tmp, statsOut] = anovan(data(:),{subjectLabels mapLabelsText}, 'varnames', {'subject', 'map'});
figure;
[results] =multcompare(statsOut, 'Dimension', [2]);
handle = gcf;
axObjs = handle.Children;
dataObjs = axObjs.Children;
for n=1:2:2*length(mapNames_extended)-1; means((n+1)/2)=dataObjs(n+1).XData; CIs(((n+1)/2),:)=dataObjs(n).XData; end
means = fliplr(means);
CIs = flipud(CIs);

figure; plot(means, 'ok', 'MarkerFaceColor', 'k'); %plot(mean_NC,'_', 'MarkerSize', 10, 'LineWidth', 4, 'MarkerFaceColor' ,'k');
hold on;
for n = 1:length(mapNames_extended)
    h = line([n n], [CIs(n,1) CIs(n,2)]);
    get(h)
    h.Color = col_ex{n};
    h.LineWidth = 2;
    set(h)
end
title(['Between ROI Model']);
%axis([0.5 length(mapNames)+0.5 0 8]);
xlabel(['Visual Field Maps'])
ylabel(['Correlation between MDS and Cortical Surface Distances'])
axis([0.5 length(mapNames_extended)+0.5 0 0.6]);
axis square
xticks([1:length(mapNames_extended)]);
xticklabels(mapNames_extended);
xtickangle(45);

