function sample(arg1, arg2)
files = dir('Class1');
% disp(files);
for file = files'
    disp(file);
    
end
% disp(arg1);
boxImage = imread('data/Class12/N2_21.jpg');
% figure;
% imshow(boxImage);
% title('Image of a Box');
sceneImage = imread('hard_multi_3.jpg');
% figure;
% imshow(sceneImage);
% title('Image of a Cluttered Scene');
boxImage = rgb2gray(boxImage);
sceneImage = rgb2gray(sceneImage);
boxPoints = detectSURFFeatures(boxImage);
scenePoints = detectSURFFeatures(sceneImage);
% figure;
% imshow(boxImage);
% title('100 Strongest Feature Points from Box Image');
% hold on;
% plsot(selectStrongest(boxPoints, 100));
% figure;
% imshow(sceneImage);
% title('300 Strongest Feature Points from Scene Image');
% hold on;
% plot(selectStrongest(scenePoints, 300));
[boxFeatures, boxPoints] = extractFeatures(boxImage, boxPoints);
[sceneFeatures, scenePoints] = extractFeatures(sceneImage, scenePoints);
boxPairs = matchFeatures(boxFeatures, sceneFeatures);
matchedBoxPoints = boxPoints(boxPairs(:, 1), :);
matchedScenePoints = scenePoints(boxPairs(:, 2), :);
figure;
showMatchedFeatures(boxImage, sceneImage, matchedBoxPoints, ...
    matchedScenePoints, 'montage');
title('Putatively Matched Points (Including Outliers)');

disp(matchedBoxPoints.Count);
disp(matchedScenePoints.Count);
if matchedBoxPoints.Count == 0
    disp('Not enough points');
%     exit;
end
[tform, inlierBoxPoints, inlierScenePoints, status] = ...
    estimateGeometricTransform(matchedBoxPoints, matchedScenePoints, 'affine');
disp(status);
disp(inlierBoxPoints.Count);
figure;
showMatchedFeatures(boxImage, sceneImage, inlierBoxPoints, ...
    inlierScenePoints, 'montage');
title('Matched Points (Inliers Only)');
% boxPolygon = [1, 1;...                           % top-left
%         size(boxImage, 2), 1;...                 % top-right
%         size(boxImage, 2), size(boxImage, 1);... % bottom-right
%         1, size(boxImage, 1);...                 % bottom-left
%         1, 1];
% newBoxPolygon = transformPointsForward(tform, boxPolygon);
% figure;
% imshow(sceneImage);
% hold on;
% line(newBoxPolygon(:, 1), newBoxPolygon(:, 2), 'Color', 'y');
% title('Detected Box');
