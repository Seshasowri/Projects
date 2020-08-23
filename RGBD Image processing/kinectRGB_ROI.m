clear
clc
close all

maxNumReg = 255; %maximum number of detectable regions, recommended to set 
                 %well above expected value of regions
minRegSize=1; %minimum size for a region to be detected (in pixels)
rMin=97; %min and max RGB values for desired region
rMax=185; %a more robust solution would be to check ratio between 
gMin=29; %RGB channels
gMax=185;
bMin=25;
bMax=185;

fcn = 0; %Set fcn=0 for recording and processing
         %Set fcn=1 for just recording data
         %Set fcn=2 for processing pre-recorded data

if(fcn==0 || fcn==1) %connects and starts the camera
    colorDevice = imaq.VideoDevice('kinect',1,'BGR_1920x1080');
    depthDevice = imaq.VideoDevice('kinect',2,'Depth_512x424');

    step(colorDevice); %inits device
    step(depthDevice);

    colorImage = step(colorDevice); %capture color and depth image
    depthImage = step(depthDevice);
    
    COLS=640;
    ROWS=480;
end

try

    if(fcn==0 || fcn==1) %create point cloud
        ptCloud = pcfromkinect(depthDevice,depthImage,colorImage);
        ptCloud = removeInvalidPoints(ptCloud); %%Optional, but will need
                                                %to export and reimport 
                                                %ptCloud
        if(fcn == 1)
            pcwrite(ptCloud,'OctArmView.ply'); %I suggest changing file 
                                               %name to automatically 
                                               %include time will produce a
                                               %more unique filename
            error('Skip to end of file'); %nothing else needs to be done 
                                          %but to disconnect camera
        end
    elseif(fcn==2)%read in point cloud
        ptCloud = pcread('OctArmView.ply');
    end
    
    cImage = ptCloud.Color;
     figure
     imshow(colorImage) %%for debugging to verify camera works and sees
     %objects
    
    location = ptCloud.Location; %ptCloud.Location is read only, 
                                 %manipulate externally
    

     %for when cImage is in vector format, with unknown dimensions
    mask = ((cImage(:,1)>=rMin) & (cImage(:,1)<=rMax) & ...
            (cImage(:,2)>=gMin) & (cImage(:,2)<=gMax) & ...
            (cImage(:,3)>=bMin) & (cImage(:,3)<=bMax));
    
    
    %mask out unneeded RGB pixels
    maskedRgbImage = bsxfun(@times, cImage, cast(mask, 'like', cImage)); 


    
    
    maskedLoc1 = bsxfun(@times,location(:,1),cast(mask,'like',location(:,1)));
    maskedLoc2 = bsxfun(@times,location(:,2),cast(mask,'like',location(:,2)));
    maskedLoc3 = bsxfun(@times,location(:,3),cast(mask,'like',location(:,3)));
    
    
    player = pcplayer(ptCloud.XLimits,ptCloud.YLimits,ptCloud.ZLimits,...
        'VerticalAxis','y','VerticalAxisDir','down');

    ptCloud.Color=maskedRgbImage; %reassign extracted color values

    view(player,ptCloud); %view identified pixels, should be only colored pixels

    xlabel(player.Axes,'X (m)');
    ylabel(player.Axes,'Y (m)');
    zlabel(player.Axes,'Z (m)');

    %extract points of interest from all points
    extracted=[maskedLoc1(mask) maskedLoc2(mask) maskedLoc3(mask)];
    labels=zeros(length(extracted),1);
    
    TotalRegions=0;
    Region=zeros(maxNumReg,3); %preallocate maximum number of regions
    
    for r=1:length(extracted)
       
       if labels(r) ~= 0
            continue;
        end
  
        sum=0;	
        for r2=-2:2
            if(r+r2>=1 && r+r2 <=length(extracted))
                sum=sum+labels(r+r2);
            end
        end

        if sum == 0  	% condition for seeding a new region is zero sum
      
          fprintf('New region at x= %f y=%f z=%f\n', num2str(extracted(r,1)),num2str(extracted(r,2)) ,num2str(extracted(r,3)));
            TotalRegions=TotalRegions+1;
            if (TotalRegions == maxNumReg)
                disp('Segmentation incomplete.  Ran out of labels.');
                break;
            end
      
            [labels, RegionSize,indices,center] = RegionGrowBrute(extracted,labels,r,nan,TotalRegions);
            if (RegionSize < 1)
        	erase region (relabel pixels back to 0)
                for i=1:RegionSize
                    labels(indices(i))=0;
                end
               TotalRegions=TotalRegions-1;
            
            else
                fprintf('Region labeled %d is %d pixels in size\n',TotalRegions,RegionSize);
                Region(TotalRegions,:)=center(:);
            end
        end
    end


    fprintf('%d total regions were found\n',TotalRegions);
    for g=1:TotalRegions
        fprintf('Region %d at (x,y,z)=(%f,%f,%f)\n',g,Region(g,1),Region(g,2),Region(g,3));
    end
    
    figure
    hold on
    scatter3(extracted(:,1),extracted(:,2),extracted(:,3),1,'o');
    %scatter3(Region(1:TotalRegions,1),Region(1:TotalRegions,2),Region(1:TotalRegions,3),100,'*r');
    axis equal
%     maskedDepthMat=reshape(,480,640);
    
catch ME %catches exceptions so that camera can be disconnected properly
     disp('Exception thrown!\n');
    disp(ME)
    disp('hello world')
end

%%Must run, even if file fails
if(fcn==0 || fcn==1)
    disp('Releasing Kinect')
    release(colorDevice);
    release(depthDevice);
    delete(colorDevice);
    delete(depthDevice);
end