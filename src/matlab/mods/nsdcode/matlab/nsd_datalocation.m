function f = nsd_datalocation(dir0)

% function f = nsd_datalocation(dir0)
%
% <dir> (optional) is [] | 'betas' | 'timeseries' | 'stimuli'
%
% Return full path to the nsddata directories.
% Edit this to suit your needs!

if ~exist('dir0','var') || isempty(dir0)
  f = 'D:\\Documents_D\\Personal\\UU\\Thesis\\CM\\nsddata\\'
  % f = '/media/charesti-start/data/NSD/nsddata/';
else
  switch dir0
  case 'betas'
    % f = '/media/charesti-start/data/NSD/nsddata_betas/';
    f = 'D:\\Documents_D\\Personal\\UU\\Thesis\\CM\\nsddata_betas\\'
  case 'timeseries'
    % f = '/media/charesti-start/data/NSD/nsddata_timeseries/';
    f = 'D:\\Documents_D\\Personal\\UU\\Thesis\\CM\\nsddata_timeseries\\'
  case 'stimuli'
    % f = '/media/charesti-start/data/NSD/nsddata_stimuli/';
    f = 'D:\\Documents_D\\Personal\\UU\\Thesis\\CM\\nsddata_stimuli\\'
  end
end
