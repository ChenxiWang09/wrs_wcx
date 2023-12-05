clc;
clear;
for i=1:4
    for j=1:8
        query_path=['/home/hlabbunri/Desktop/wcx/wrs-wcx/wcx/fdcmwork/data/result/0124/rotate_data/',int2str(i),'_',int2str(j),'.pgm'];
        template_path=['/home/hlabbunri/Desktop/wcx/wrs-wcx/wcx/fdcmwork/data/temp/temp_outline/',int2str(j),'.pgm'];
        x1=48; y1=60; x2=140; y2=248;
        threshold_of_size=1;
        [detection]=chamfermatching_nopic_forpython(query_path,template_path, x1,x2,y1,y2,threshold_of_size);
    end
end