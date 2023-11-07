function [indata,filenames] = Dataload_Extracellular(search_term,layers)
% This function loads all .mat files in the current folder based on the
% search criteria provided. The number of layers corresponds to the
% number of layers used for the visual stimulus. 
% Unlike Dataload_RawData it does not load the undiscriminated dataset
filelist = dir(fullfile('230214_N01P01_TSDN', search_term));
filenames = {filelist.name};
nfiles = length(filenames);
indata = cell(nfiles,1);

for K = 1:nfiles
if layers==1
indata{K} = load(filenames{K},'DataBlock', 'Units','Layer_1_Parameters','debugData');
elseif layers==2
indata{K} = load(filenames{K},'DataBlock', 'Units','Layer_1_Parameters','Layer_2_Parameters','debugData','fileName');    
elseif layers==3
indata{K} = load(filenames{K},'DataBlock', 'Units','Layer_1_Parameters','Layer_2_Parameters','Layer_3_Parameters','debugData');
elseif layers==4
indata{K} = load(filenames{K},'DataBlock', 'Units','Layer_1_Parameters','Layer_2_Parameters','Layer_3_Parameters','Layer_4_Parameters','debugData');       
else
disp('Too many layers. Modify script')
end        
end
end

