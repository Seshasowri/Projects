colorDevice = imaq.VideoDevice('kinect',1,'BGR_1920x1080');
depthDevice = imaq.VideoDevice('kinect',2,'Depth_512x424');

step(colorDevice);
step(depthDevice);

colorImage = step(colorDevice);
depthImage = step(depthDevice);
    
imshow(colorImage);

disp('Releasing Kinect')
release(colorDevice);
release(depthDevice);
delete(colorDevice);
delete(depthDevice);