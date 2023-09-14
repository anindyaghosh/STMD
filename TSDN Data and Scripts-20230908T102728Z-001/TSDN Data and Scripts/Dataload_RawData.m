function [indata,filenames] = Dataload_RawData(search_term,layers,filename_toload)
% This function loads only one of the .mat files in the current folder 
% based on the search criteria provided and the number listed as the 
% filename_toload. The number of layers corresponds to the number of layers 
% used for the visual stimulus.
% Unlike Dataload_Extracellular this function includes the raw data trace 
% in the information loaded.

filelist = dir(search_term);
filenames = {filelist.name};
nfiles = length(filenames);
indata = cell(nfiles,1);
K = filename_toload;

if layers==1
indata{1} = load(filenames{K},'DataBlock', 'Units','Layer_1_Parameters','debugData');
elseif layers==2
indata{1} = load(filenames{K},'DataBlock', 'Units','Layer_1_Parameters','Layer_2_Parameters','debugData','fileName');    
elseif layers==3
indata{1} = load(filenames{K},'DataBlock', 'Units','Layer_1_Parameters','Layer_2_Parameters','Layer_3_Parameters','debugData');
elseif layers==4
indata{1} = load(filenames{K},'DataBlock', 'Units','Layer_1_Parameters','Layer_2_Parameters','Layer_3_Parameters','Layer_4_Parameters','debugData');       
         
else
disp('Too many layers. Modify script')
end        
end

