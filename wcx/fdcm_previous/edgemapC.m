%generate pgm file

clc;
clear;
Input_path ='C:\Users\wangq\PycharmProjects\pythonProject\intelrealsense_wcx\templateimage\newtemplate\';  
Output_path='C:\Users\wangq\PycharmProjects\pythonProject\intelrealsense_wcx\templateimage\newtemplate\';
namelist = dir(strcat(Input_path,'*.png'));  %����ļ��������е� .jpgͼƬ
len = length(namelist);
    for i = 1:len
   
    name=namelist(i).name;  %namelist(i).name; %�����õ�ֻ�Ǹ�·���µ��ļ���
    I=imread(strcat(Input_path, name)); %ͼƬ������·����
    % I=imread('C:\Users\qiumian\Desktop\mastercourse\research\homography transformation\data\part\part1\trasnformed\rotation within paper\maxpool\rotate1.jpg.jpg');
     BW0=rgb2gray(I);
%      BW1=DUCO_RemoveBackGround(	BW0,4,0);
% imwrite(BW0,[Output_path,name,'.jpg']);
%     BW1=trans(I);
%     BW1 = imadjust(BW0);
     BW1= edge(BW0,'canny',[0.02,0.04]);  % ����canny����
%        figure(1)
%       subplot(2,2,1);
       imshow(BW1,[]);
%       subplot(2,2,2);
%       imshow(BW1,[]);
%       subplot(2,2,3);
%       imshow(BW2,[]);
%       pause;    
      imwrite(BW1,[Output_path,name,'.pgm']); %������ͼƬ�洢��·����  �������ε����� 
  end
