% CS194-26 (cs219-26): Project 1, starter Matlab code
clc
clear
close all
% name of the input file
namelist = dir(['.\data\data\','*.jpg']);
for kk=1:1:size(namelist,1)
    
    path = strcat(namelist(kk).folder, '\' ,namelist(kk).name);
    imname = namelist(kk).name;
    % read in the image
    fullim = imread(path);
    
    % convert to double matrix (might want to do this later on to same memory)
    fullim = im2double(fullim);
    
    % compute the height of each part (just 1/3 of total)
    height = floor(size(fullim,1)/3);
    % separate color channels
    B = fullim(1:height,:);
    G = fullim(height+1:height*2,:);
    R = fullim(height*2+1:height*3,:);
    
    % Align the images
    % Functions that might be useful to you for aligning the images include:
    % "circshift", "sum", and "imresize" (for multiscale)
    %%%%%aG = align(G,B);
    %%%%%aR = align(R,B);
    %for G
    if size(B,1)>300 && size(B,2)>300
        B =imresize(B,[300,300]);
        G =imresize(G,[300,300]);
        R =imresize(R,[300,300]);
    end
    %去除边缘 自动分割
    figure(1)
    bw=edge(B,'sobel');
    bw=bwareaopen(bw, 200); %后面那个值是减少的面积 这样得到就是一个边框的线
    imshow(bw)
    flag=0;
    %获取边缘属性
    for i=300:-1:1
        for j=1:1:300
            if bw(i,j)==1
                bb(1)=i;
                bb(2)=j;
                flag=1;
                break
                
            end
        end
        if flag==1
            break
        end
    end
    flag=0;
    for i=1:1:300
        for j=300:-1:1
            if bw(i,j)==1
                bb(3)=i;
                bb(4)=j;
                flag=1;
                break
            end
        end
        if flag==1
            break
        end
    end
    
    B = B(floor(bb(3))+5 :floor(bb(1))-5 ,floor(bb(2))+5:floor(bb(4))-5);
    figure(2)
    imshow(G)
    G = G(floor(bb(3))+5 :floor(bb(1))-5 ,floor(bb(2))+5:floor(bb(4))-5) ;
    R = R(floor(bb(3))+5:floor(bb(1))-5,floor(bb(2))+5:floor(bb(4))-5) ;
    % 调整对比度 可选
    B=imadjust(B);
    G=imadjust(G);
    R=imadjust(R);
    
    %获取最佳对其 x 和 y方向
    count=1;
    for i=-30:1:30 %x方向
        for j=-30:1:30 %y方向
            G_new=circshift(G,[i j]);
            r(count,1)=i;
            r(count,2)=j;
            r(count,3)=sum(sum((G_new-B).^2));
            count=count+1;
        end
    end
    [~,index]=min(r(:,3));
    aG=circshift(G,[r(index,1) r(index,2)]);
    % for R
    count=1;
    for i=-30:1:30
        for j=-30:1:30
            R_new=circshift(R,[i j]);
            r1(count,1)=i;
            r1(count,2)=j;
            r1(count,3)=sum(sum((R_new-B).^2));
            count=count+1;
        end
    end
    [~,index]=min(r1(:,3));
    aR=circshift(R,[r1(index,1) r1(index,2)]);
    
    % open figure
    %% figure(1);
    colorim=cat(3,aR,aG,B);
    figure(3)
    imshow(colorim)
    % create a color image (3D array)
    % ... use the "cat" command
    % show the resulting image
    % ... use the "imshow" command
    % save result image
    %% imwrite(colorim,['result-' imname]);
    imwrite(colorim,['result-' imname])
end