

%Option set
optionStruct.toleranceLevel=0; %Tolerance for convergence
optionStruct.numSeeds=1; %Number of seeds
optionStruct.waitBarOn=0; %Turn on/off waitbar

subjs = {1, 2, 3, 4, 5, 6, 7 , 8};
%subjs = {2, 3, 4, 5, 7};
hemis = {'lh', 'rh'};
rois = {
    [1 2]          % V1v+d
    3              % V2v
    4              % V2d
    5              % V3v
    6              % V3d
    7              % hV4
    8              % VO-1
    9              % VO-2
    10             % PHC-1
    11             % PHC-2
    12             % LO-1
    13             % LO-2
    14             % TO-1
    15             % TO-2
};
roi_names = {
    'V1'
    'V2v'
    'V2d'
    'V3v'
    'V3d'
    'hV4'
    'VO-1'
    'VO-2'
    'PHC-1'
    'PHC-2'
    'LO-1'
    'LO-2'
    'TO-1'
    'TO-2'
};

% dir where roi files are and voxel pairs distances will be stored
% change this based on your working directory 
datadir = '/media/Working/stan-thesis/data/nsddata/freesurfer/';
% dir where the surf files are, (nsddata)
surfdir = cvnpath('freesurfer');

for zz=1:length(subjs)
    subjid = sprintf('subj%02d',subjs{zz});
    for hh=1:length(hemis)
        hemi = hemis{hh};
        if zz == 6 | zz == 8
            hhroivals = cvnloadmgz(sprintf('%s/%s/label/customrois/%s.%s.nans_del.testrois.mgz',datadir,subjid,hemi,subjid)); % nans_del is the mask where the nans in the betas are deleted
        else
            hhroivals = cvnloadmgz(sprintf('%s/%s/label/customrois/%s.%s.testrois.mgz',datadir,subjid,hemi,subjid));
        end 
        nVoxels = size(hhroivals,1);
        if zz == 6 | zz == 8
            white_file = sprintf('%s/%s/surf/%s.white_del', surfdir, subjid, hemi);
        else
            white_file = sprintf('%s/%s/surf/%s.white', surfdir, subjid, hemi);
        end
        [V, F, t]  = freesurfer_read_surf_kj(white_file, 'verbose', true);
        for rr=1:length(rois)
            roi = rois{rr};
            roi_name = roi_names{rr};
            distances_dir = sprintf('%s/%s/label/distances', datadir, subjid);
            if ~exist(distances_dir, 'dir')
               mkdir(distances_dir)
            end
            d_roi_file = sprintf('%s/%s.%s.dists.%s.mat', distances_dir, hemi, subjid, roi_name);
            fprintf('enter %s:%s.%s\n', subjid, hemi, roi_name);
            if ~isfile(d_roi_file)
                vroi = find(ismember(hhroivals, roi), inf, 'first');
                nVoxelsHhRoi = size(vroi, 1);
                d_roi = nan(nVoxelsHhRoi, nVoxelsHhRoi);
                hw=waitbar(1/nVoxelsHhRoi,[sprintf('%s.%s:[%s] Voxels distances...',hemi, subjid, roi_name), num2str(round(100.*1/nVoxelsHhRoi)),'%']);
                % get pairs of distances
                for vv=1:nVoxelsHhRoi
                    waitbar(vv/nVoxelsHhRoi,hw,[sprintf('%s.%s:[%s] Voxels distances...',hemi, subjid, roi_name), num2str(round(100.*vv/nVoxelsHhRoi)),'%']);
                    indStart = vroi(vv);
                    d_roi(:,vv) = meshDistMarchModded(F, V, indStart, optionStruct, vroi);
                end
                close(hw);
                % convergence from different datapoints may have differ in
                % floating point precision so just take mins
                d_roi_tu = triu(d_roi)';
                d_roi_tl = tril(d_roi);
                d_roi_tmin = min(d_roi_tu, d_roi_tl);
                d_roi = d_roi_tmin + d_roi_tmin';
                save(d_roi_file, 'd_roi');
            end
            fprintf('cleared %s:%s.%s\n', subjid, hemi, roi_name);
        end
    end
end