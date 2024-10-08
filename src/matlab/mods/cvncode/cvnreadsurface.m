function varargout = cvnreadsurface(subject, hemi, surftype, surfsuffix, varargin)
% surface_or_count = cvnreadsurface(subject, hemi, surftype, surfsuffix, 'param',value,...)
%
% Read a freesurfer surface (verts,faces) from file, or just the vertex count
%
% Inputs:
%   subject:               Freesurfer subject
%   hemi:                  'lh','rh', or {'lh','rh'}
%   surftype:              sphere|inflated|white|pial|layerA1 ...
%   surfsuffix (optional): DENSE|DENSETRUNCpt|orig (default) ("orig"=<hemi>.sphere)
%
% Outputs:
%   surface_or_count:   a struct for each hemisphere, containing
%                       'vertices': Nx3 vertex xyz coords
%                       'faces': Fx3 vertex indices
%        (for patches): 'patchmask': Nx1 logical if vertex belongs to patch
%
% Optional inputs:  'paramname','value',...
%   justcount:          true or false.  If true, return vertex count only  (default=false)
%   surfdir:            (default = <cvnpath('freesurfer')>/<subject>/surf)
%   noscale:            true or false. If true, no patch rescaling from orig->white (default=false)
%
% Examples:
%   >> leftN=cvnreadsurface('C0041', 'lh', 'sphere', 'DENSETRUNCpt','justcount',true)
%   leftN =
%       412008
%
%   >> [leftN rightN]=cvnreadsurface('C0041', {'lh','rh'}, 'sphere', 'DENSETRUNCpt','justcount',true)
%   leftN =
%       412008
%   rightN =
%       469547
%
%   >> surfL=cvnreadsurface('C0041', 'lh', 'sphere', 'DENSETRUNCpt')
%  surfL = 
%      vertices: [412008x3 double]
%         faces: [821933x3 double]
%
%  >> [surfL surfR]=cvnreadsurface('C0041', {'lh','rh'}, 'sphere', 'DENSETRUNCpt')
%  surfL = 
%      vertices: [412008x3 double]
%         faces: [821933x3 double]
%  surfR = 
%      vertices: [469547x3 double]
%         faces: [937086x3 double]
%

% Update KJ 2017-08-03: Allow patch surface types

% inputs
if ~exist('surfsuffix','var') || isempty(surfsuffix)
  surfsuffix = 'orig';
end

%%%%%%%%%%%%%%%%%%%%
%default options
options=struct(...
    'justcount',false,...
    'noscale',false,...
    'surfdir',[]);

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
%parse options
input_opts=mergestruct(varargin{:});
fn=fieldnames(input_opts);
for f = 1:numel(fn)
    opt=input_opts.(fn{f});
    if(~(isnumeric(opt) && isempty(opt)))
        options.(fn{f})=input_opts.(fn{f});
    end
end
%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
surfdir=[];
if(~isempty(options.surfdir) && exist(options.surfdir,'dir'))
    surfdir=options.surfdir;
else
    freesurfdir=cvnpath('freesurfer');
    % CHANGED / FOR \\
    surfdir=sprintf('%s\\%s\\surf',freesurfdir,subject);
end

assert(exist(surfdir,'dir')>0);

if(isempty(hemi))
    hemi={'lh','rh'};
elseif(ischar(hemi))
    hemi={hemi};
end

if(strcmpi(surfsuffix,'orig'))
    suffix_file='';
else
    suffix_file=surfsuffix;
end

result={};

for h = 1:numel(hemi)

    
    patchfile=[];
    if(regexpmatch(surftype,'\.patch\.'))
        suffix2=suffix_file;
        if(~isempty(suffix_file))
            suffix2=[suffix2 '.'];
        end
        % CHANGED / FOR \\
        patchfile=sprintf('%s\\%s.%s%s',surfdir,hemi{h},suffix2,surftype);
        assert(exist(patchfile,'file')>0,'file not found: %s',patchfile);
        
        surffile=sprintf('%s/%s.%s%s',surfdir,hemi{h},'white',suffix_file);
    else
        surffile=sprintf('%s/%s.%s%s',surfdir,hemi{h},surftype,suffix_file);
    end
    
    assert(exist(surffile,'file')>0,'file not found: %s',surffile);
    
    
    if(options.justcount)
        vertsN = freesurfer_read_surf_kj(surffile,'justcount',true);
        result{h}=vertsN;
    else
        [verts,faces] = freesurfer_read_surf_kj(surffile);
        patchmask=[];
        if(~isempty(patchfile))
            origverts=verts;
        
            %1. read in patch file
            %2. identify faces that contain 3 patch
            %3. insert patch vertex locations into full surface vertices
            pstruct=fast_read_patch_kj(patchfile);
            patchmask=false(size(verts,1),1);
            patchmask(pstruct.vno)=true;
            faces=faces(sum(patchmask(faces),2)==3,:);
            verts(pstruct.vno,:)=[pstruct.x(:) pstruct.y(:) pstruct.z(:)];
            

            if(regexpmatch(surftype,'\.flat\.'))
                % If *.flat.patch.*, do extra 2D polygon-based cleanup to
                % detect funny outlier edge vertices and remove faces that
                % contain those bad vertices
                [faces,patchmask] = cleanflatpatch(faces,verts);
            end
            
            %%%%% rescale patch vertices to match anatomical surface size
            if(~options.noscale)
                %1. compute patch surface area on ?h.white ("true anatomical area")
                Aorig=sum(facearea(faces,origverts));

                %2. compute patch surface area on flat (which should be close
                %   to area on smoothwm, ie smaller than true anatomical surface)
                Aflat=sum(facearea(faces,verts));

                %scale patch vertices so that final surface area matches the ?h.white surface area
                flatverts=verts(patchmask,:);
                flatverts_mean=mean(flatverts,1);
                verts(patchmask,:)=bsxfun(@plus,bsxfun(@minus,flatverts,flatverts_mean)*sqrt(Aorig/Aflat),flatverts_mean);
            end
        end
        result{h}=struct('vertices',verts,'faces',faces);
        if(~isempty(patchmask))
            result{h}.patchmask=patchmask;
        end
    end
end

if(numel(result) > 1 && nargout==1)
    varargout={result};
else
    varargout=result;
end