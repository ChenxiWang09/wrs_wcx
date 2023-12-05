function output = changechannel( input )
    %matlab image的图像通道顺序为R，G，B
    %opencv image的图像通道顺序为B，G，R
    %此函数进行1,3通道的交换
    output=input;
    output(:,:,1)=input(:,:,3);
    output(:,:,3)=input(:,:,1);
end