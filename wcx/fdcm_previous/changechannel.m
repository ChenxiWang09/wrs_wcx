function output = changechannel( input )
    %matlab image��ͼ��ͨ��˳��ΪR��G��B
    %opencv image��ͼ��ͨ��˳��ΪB��G��R
    %�˺�������1,3ͨ���Ľ���
    output=input;
    output(:,:,1)=input(:,:,3);
    output(:,:,3)=input(:,:,1);
end