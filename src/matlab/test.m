zz = 1;
subjid = sprintf('subj%02d',zz); 

curv = cvnreadsurfacemetric(subjid,{'lh','rh'},'curv',[],'orig'); 
curv