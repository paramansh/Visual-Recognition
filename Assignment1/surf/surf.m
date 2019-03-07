function surf(arg1, arg2)
Imagedir = dir('Images/Class*/');
for folders = Imagedir'
    disp(folders.name)
    files = dir(strcat('Images/', folders.name, '/*.jpg'));
    % disp(files);
    max_matches = 0;
    max_file = '';
    for file = files'
%         disp(file.name);
        boxImage = imread(strcat('Images/',folders.name, '/', file.name));

        sceneImage = imread('hard_multi_3.jpg');

        boxImage = rgb2gray(boxImage);
        sceneImage = rgb2gray(sceneImage);
        boxPoints = detectSURFFeatures(boxImage);
        scenePoints = detectSURFFeatures(sceneImage);

        [boxFeatures, boxPoints] = extractFeatures(boxImage, boxPoints);
        [sceneFeatures, scenePoints] = extractFeatures(sceneImage, scenePoints);
        boxPairs = matchFeatures(boxFeatures, sceneFeatures);
        matchedBoxPoints = boxPoints(boxPairs(:, 1), :);
        matchedScenePoints = scenePoints(boxPairs(:, 2), :);


    %     disp(matchedBoxPoints.Count);
    %     disp(matchedScenePoints.Count);
        if matchedBoxPoints.Count == 0
%             disp('Not enough points');
    %         exit;
        else [tform, inlierBoxPoints, inlierScenePoints, status] = ...
            estimateGeometricTransform(matchedBoxPoints, matchedScenePoints, 'affine');

    %         disp(inlierBoxPoints.Count);
            if (inlierBoxPoints.Count > max_matches)
                max_matches = inlierBoxPoints.Count;
                max_file = file.name;
            end
        end
    end

    disp(max_file);
    disp(max_matches);
end