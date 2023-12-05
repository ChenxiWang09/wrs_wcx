
function [detection]=chamfermatching_nopic_forpython(query_path,template_path, x1,x2,y1,y2,threshold_of_size)

%//==================================================================
%(1)input parameter 
% query_path: path of query edge image, format:pgm.
% template_path: path of template edge image, format:pgm
% candidate_amount:the amount of candidate
% x1,y1,x2,y2: the roughly position of object in the query image
% threshold_of_size is the parameter of the error limitation of image position
%(2)output parameter
%partname: detected partname for limit amount
%relative cost value
%//==================================================================

query=imread(query_path);
disp('Fast Directional Chamfer Matching for searching matching');
%//==================================================================
%// Basic Configuration
%//==================================================================

sz=size(query);
threshold = 0.5;
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

infor = zeros(1, 7);
infor(:,5)=10;

templateEdgeMap=imread(template_path);
% sz=size(templateEdgeMap);
% for i = 1:sz[0]
%     for j = 1:sz[1]
%         if templateEdgeMap(i,j) == 255
%             templateEdgeMap(i,j) = 1;
%         end
%     end
% end

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
infor(1,7)=1;
% position check
for j=1:nDetection
    if(abs(detWinds(j,1)-x1)<(sz(2)/threshold_of_size) && abs(detWinds(j,2)-y1)<(sz(1)/threshold_of_size) && abs(detWinds(j,1) + detWinds(j,3)-x2)<(sz(2)/threshold_of_size)  && abs(detWinds(j,2)+detWinds(j,4)-y2)<(sz(1)/threshold_of_size))
        infor(1,1:6) = detWinds(j,:);
        break;
    end
end
if(cost~=1)
    color = [0 1 0];
    lineWidth = 3;
    imshow(queryColor,[]);
    sx = detWinds(j,1);
    ex = sx + detWinds(j,3);
    sy = detWinds(j,2);
    ey = sy + detWinds(j,4);
    line([sx ex],[sy sy],'Color',color,'LineWidth',lineWidth);
    line([sx ex],[ey ey],'Color',color,'LineWidth',lineWidth);
    line([sx sx],[sy ey],'Color',color,'LineWidth',lineWidth);
    line([ex ex],[sy ey],'Color',color,'LineWidth',lineWidth);
    text(sx,sy-10,num2str(cost),'Color','yellow','FontSize',14);
end
detection = infor;


end
