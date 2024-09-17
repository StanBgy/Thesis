function CCmax = compute_CCmax(Rns,voxel_mask)

N = length(Rns);

selected_Rns = cellfun(@(x) x(:,voxel_mask), Rns, 'UniformOutput', false);
var_Rns = cellfun(@var, selected_Rns, 'UniformOutput', false);
reshaped_var_Rns = reshape(cell2mat(var_Rns'),N,sum(voxel_mask));
factor_top = (N-1)*sum(reshaped_var_Rns,1);

reshaped_selected_Rns = reshape(cell2mat(selected_Rns),size(selected_Rns{1},1),size(selected_Rns{1},2),N);
sum_Rns = sum(reshaped_selected_Rns,3);
factor_bottom = var(sum_Rns,0,1) - sum(reshaped_var_Rns,1);
CCmax = 1./(sqrt(1+(1/N)*(factor_top./factor_bottom-1)));

CCmax(imag(CCmax)==0) = real(CCmax(imag(CCmax)==0));
CCmax(imag(CCmax)~=0) = 0;

% sometimes CCmax becomes imaginary
% I think this is when the noise power is higher than the signal power
% I now set the max VE there to 0
% how many?