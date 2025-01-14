
function [partname,value]=chamfermatching_nopic_forpython(queryedgeaddress,tempedgeaddress,x0,y0,x1,y1,times)
query=imread(queryedgeaddress);
disp('Fast Directional Chamfer Matching for searching matching');
%//==================================================================
%// Basic Configuration
%//==================================================================
Input_path = 'C:\Users\wangq\PycharmProjects\pythonProject\data\2022.12.15\template\';  %template path
Output_path='C:\Users\wangq\PycharmProjects\pythonProject\data\2022.12.15\result\';
namelist_pgm = dir(strcat(Input_path,'*.pgm'));   
len = length(namelist_pgm);


sz=size(query);
threshold = 1;

value=1;
partname='7';
% Set the parameter for line fitting function
lineMatchingPara = struct(...
    'NUMBER_DIRECTION',60,...
    'DIRECTIONAL_COST',0.5,...
    'MAXIMUM_EDGE_COST',30,...
    'MATCHING_SCALE',1.0,...
    'TEMPLATE_SCALE',0.6761,...
    'BASE_SEARCH_SCALE',1.20,...
    'MIN_SEARCH_SCALE',-7,...
    'MAX_SEARCH_SCALE',0,...
    'BASE_SEARCH_ASPECT',1.1,...
    'MIN_SEARCH_ASPECT',-1,...
    'MAX_SEARCH_ASPECT',1,...    
    'SEARCH_STEP_SIZE',2,...
    'SEARCH_BOUNDARY_SIZE',2,...
    'MIN_COST_RATIO',1.0...    
    );
% Set the parameter for line fitting function
lineFittingPara = struct(...
    'SIGMA_FIT_A_LINE',0.5,...
    'SIGMA_FIND_SUPPORT',0.5,...
    'MAX_GAP',2.0,...
    'N_LINES_TO_FIT_IN_STAGE_1',300,...
    'N_TRIALS_PER_LINE_IN_STAGE_1',100,...
    'N_LINES_TO_FIT_IN_STAGE_2',100000,...
    'N_TRIALS_PER_LINE_IN_STAGE_2',1);

lineFittingPara2 = struct(...
    'SIGMA_FIT_A_LINE',0.5,...
    'SIGMA_FIND_SUPPORT',0.5,...
    'MAX_GAP',100,...
    'N_LINES_TO_FIT_IN_STAGE_1',0,...
    'N_TRIALS_PER_LINE_IN_STAGE_1',0,...
    'N_LINES_TO_FIT_IN_STAGE_2',100000,...
    'N_TRIALS_PER_LINE_IN_STAGE_2',1);


templateEdgeMap=imread(tempedgeaddress);
%//==================================================================
%// Convert edge map into line representation
%//==================================================================

% convert the template edge map into a line representation
[lineRep lineMap] = mex_fitline(double(templateEdgeMap),lineFittingPara);
% display the top few line segments to illustrate the representation
nLine = size(lineRep,1);
%//==================================================================
%// FDCM detection
%//==================================================================
template = cell(1);
tempate{1} = lineRep;

[detWinds] = mex_fdcm_detect(double(query),tempate,threshold,...
    lineFittingPara2,lineMatchingPara);
nDetection = size(detWinds,1);
cost=1;
number=0;

for i=1:min(nDetection,2)
    if(abs(detWinds(i,1)-x0)<(sz(2)/times) && abs(detWinds(i,2)-y0)<(sz(1)/times) && abs(detWinds(i,1) + detWinds(i,3)-x1)<(sz(2)/times)  && abs(detWinds(i,2)+detWinds(i,4)-y1)<(sz(1)/times))
        if(detWinds(i,5)<cost)
            
            cost=detWinds(i,5);
            number=i;
        end
    end
end

if(cost~=1)
    imshow(query);

    color = [0 1 0];

    lineWidth = 3;
    hold on;
    i=number;
    sx = detWinds(i,1);
    ex = sx + detWinds(i,3);
    sy = detWinds(i,2);
    ey = sy + detWinds(i,4);

    line([sx ex],[sy sy],'Color',color,'LineWidth',lineWidth);
    line([sx ex],[ey ey],'Color',color,'LineWidth',lineWidth);
    line([sx sx],[sy ey],'Color',color,'LineWidth',lineWidth);
    line([ex ex],[sy ey],'Color',color,'LineWidth',lineWidth);
    text(sx,sy-10,num2str(cost),'Color','yellow','FontSize',14);
end


if(cost<value)
    value=cost;
end
partname=7;

end