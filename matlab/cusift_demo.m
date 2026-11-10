%CUSIFT_DEMO Align the two bundled test images from MATLAB.
%   Mirrors cusift_py/test.py: extract, match, estimate, warp, then the fused
%   pipeline, printing counts and timings and showing the warped canvases.

here = fileparts(mfilename('fullpath'));
img1 = imread(fullfile(here, '..', 'test', 'img1.png'));
img2 = imread(fullfile(here, '..', 'test', 'img2.png'));
if ndims(img1) == 3, img1 = rgb2gray(img1); end
if ndims(img2) == 3, img2 = rgb2gray(img2); end

sift = CuSift();
fprintf('cusift %s\n', strjoin(string(sift.version()), '.'));

% Stage by stage
t = tic;
kp1 = sift.extract(img1);
kp2 = sift.extract(img2);
fprintf('extract: %d and %d keypoints in %.1f ms\n', kp1.Count, kp2.Count, 1e3 * toc(t));

t = tic;
sift.match(kp1, kp2);
m = kp1.match();
fprintf('match:   %d of %d keypoints matched in %.1f ms\n', nnz(m), kp1.Count, 1e3 * toc(t));

t = tic;
[H, inliers] = sift.estimateHomography(kp1, sift.homographyOptions('seed', 42));
fprintf('homography: %d inliers in %.1f ms\n', inliers, 1e3 * toc(t));
disp(H);

canvas = sift.computeCanvas(H, size(img1), size(img2));
fprintf('canvas: %dx%d, origin (%d, %d)\n', canvas.size(1), canvas.size(2), canvas.origin(1), canvas.origin(2));

t = tic;
[w1, w2] = sift.warp(img1, img2, H);
fprintf('warp (GPU): %dx%d in %.1f ms\n', size(w1, 2), size(w1, 1), 1e3 * toc(t));

% The fused pipeline, reusing the device-resident images across stages
t = tic;
[kpA, kpB, H2, inliers2, wA, wB] = sift.alignAndWarp(img1, img2, [], sift.homographyOptions('seed', 42));
fprintf('alignAndWarp: %d / %d keypoints, %d inliers, %.1f ms\n', kpA.Count, kpB.Count, inliers2, 1e3 * toc(t));
fprintf('same homography as the staged run: %d\n', isequal(H, H2));

fprintf('device memory for this pair: %.1f MiB\n', sift.memoryPipeline(size(img1), size(img2)) / 2^20);

figure('Name', 'cusift');
subplot(1, 2, 1); imshow(uint8(w1)); title('image 1 on the canvas');
subplot(1, 2, 2); imshow(uint8(w2)); title('image 2 warped onto it');

% Keypoint sets free themselves when their objects are deleted; warped images
% were copied into MATLAB arrays already.  Before unloadlibrary, run
% CuSift.unload() so nothing is left behind on the GPU.
