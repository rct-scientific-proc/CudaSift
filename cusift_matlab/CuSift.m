classdef CuSift < handle
    %CUSIFT  MATLAB wrapper for the CudaSift GPU-accelerated SIFT library.
    %
    %   obj = CuSift()            — auto-detect DLL next to this .m file
    %   obj = CuSift(dllPath)     — explicit path to cusift.dll / .so
    %   obj = CuSift(dllPath, headerPath)
    %
    %   The library is loaded once via loadlibrary and shared across all
    %   instances.  Call  CuSift.unloadLib()  to unload explicitly.
    %
    %   Example (full pipeline):
    %       cs = CuSift();
    %       img1 = single(rgb2gray(imread('left.png')));
    %       img2 = single(rgb2gray(imread('right.png')));
    %       opts = cs.defaultExtractOptions();
    %       hopts = cs.defaultHomographyOptions();
    %       [h1, h2, H, nMatches] = cs.extractAndMatchAndFindHomography( ...
    %           img1, img2, opts, hopts);
    %       fprintf('Found %d matches\n', nMatches);
    %       pts = cs.getAllPoints(h1);
    %       cs.deleteSiftData(h1);
    %       cs.deleteSiftData(h2);

    properties (Constant, Access = private)
        LibName = 'cusift';
    end

    properties (Access = private)
        dllPath_    char
        headerPath_ char
    end

    % ---------------------------------
    %  Construction / library loading
    % ---------------------------------
    methods
        function obj = CuSift(dllPath, headerPath)
            %CUSIFT  Create a CuSift wrapper and load the shared library.
            %
            %   obj = CuSift()
            %   obj = CuSift(dllPath)
            %   obj = CuSift(dllPath, headerPath)

            thisDir = fileparts(mfilename('fullpath'));

            if nargin < 1 || isempty(dllPath)
                if ispc
                    dllPath = fullfile(thisDir, 'cusift.dll');
                elseif ismac
                    dllPath = fullfile(thisDir, 'libcusift.dylib');
                else
                    dllPath = fullfile(thisDir, 'libcusift.so');
                end
            end
            if nargin < 2 || isempty(headerPath)
                headerPath = fullfile(thisDir, '..', 'cusift.h');
            end

            obj.dllPath_    = dllPath;
            obj.headerPath_ = headerPath;
            obj.ensureLoaded();
        end
    end

    methods (Static)
        function unloadLib()
            %UNLOADLIB  Unload the cusift shared library.
            if libisloaded(CuSift.LibName)
                % Release all handles inside the library first
                calllib(CuSift.LibName, 'CusiftDeleteAllSiftData');
                unloadlibrary(CuSift.LibName);
            end
        end

        function tf = isLoaded()
            tf = libisloaded(CuSift.LibName);
        end
    end

    methods (Access = private)
        function ensureLoaded(obj)
            if ~libisloaded(obj.LibName)
                loadlibrary(obj.dllPath_, obj.headerPath_);
                calllib(obj.LibName, 'InitializeCudaSift');
            end
        end
    end

    % ---------------------------------
    %  Default option helpers
    % ---------------------------------
    methods (Static)
        function opts = defaultExtractOptions()
            %DEFAULTEXTRACTOPTIONS  Return an ExtractSiftOptions_t struct
            %   with sensible defaults.
            opts.thresh_                   = single(3.0);
            opts.lowest_scale_             = single(0.0);
            opts.highest_scale_            = single(inf);
            opts.edge_thresh_              = single(10.0);
            opts.init_blur_                = single(1.0);
            opts.max_keypoints_            = int32(8192);
            opts.num_octaves_              = int32(5);
            opts.scale_suppression_radius_ = single(0.0);
        end

        function opts = defaultHomographyOptions()
            %DEFAULTHOMOGRAPHYOPTIONS  Return a FindHomographyOptions_t
            %   struct with sensible defaults.
            opts.num_loops_            = int32(1000);
            opts.min_score_            = single(0.0);
            opts.max_ambiguity_        = single(1.0);
            opts.thresh_               = single(5.0);
            opts.improve_num_loops_    = int32(5);
            opts.improve_min_score_    = single(0.0);
            opts.improve_max_ambiguity_= single(1.0);
            opts.improve_thresh_       = single(3.0);
            opts.seed_                 = uint32(0);
            opts.model_type_           = int32(0);   % CUSIFT_MODEL_HOMOGRAPHY
        end
    end

    % ---------------------------------
    %  Error checking (private)
    % ---------------------------------
    methods (Access = private)
        function checkError(obj, funcName) %#ok<INUSL>
            hadErr = calllib(CuSift.LibName, 'CusiftHadError');
            if hadErr ~= 0
                linePtr = libpointer('int32Ptr', int32(0));
                fnBuf   = blanks(256);
                msgBuf  = blanks(256);
                calllib(CuSift.LibName, 'CusiftGetLastErrorString', ...
                        linePtr, fnBuf, msgBuf);
                error('CuSift:%s', funcName, ...
                      'CuSift error in %s (file %s, line %d): %s', ...
                      funcName, deblank(fnBuf), linePtr.Value, deblank(msgBuf));
            end
        end
    end

    % ---------------------------------
    %  Image helpers (private)
    % ---------------------------------
    methods (Access = private, Static)
        function imgStruct = makeImageStruct(img)
            %MAKEIMAGESTRUCT  Convert an H x W single matrix into an
            %   Image_t struct suitable for calllib.
            assert(isa(img, 'single'), 'CuSift:badType', ...
                   'Image must be single-precision (float32).');
            assert(ismatrix(img), 'CuSift:badDim', ...
                   'Image must be 2-D (grayscale). Convert with rgb2gray first.');
            [h, w] = size(img);
            % C library expects row-major; MATLAB stores column-major.
            % Transpose so that MATLAB's column-major layout becomes row-major.
            rowMajor = img.';
            imgStruct.host_img_ = libpointer('singlePtr', rowMajor(:));
            imgStruct.width_    = int32(w);
            imgStruct.height_   = int32(h);
        end
    end

    % ---------------------------------
    %  Core API wrappers
    % ---------------------------------
    methods
        % ── Extract ──────────────────────────────────────────────────
        function handle = extractSift(obj, img, extractOpts)
            %EXTRACTSIFT  Extract SIFT features from an image.
            %
            %   handle = obj.extractSift(img, extractOpts)
            %
            %   img          - H x W single matrix (grayscale, 0-255).
            %   extractOpts  - struct from defaultExtractOptions().
            %   handle       - opaque int32 handle; free with deleteSiftData().
            imgS = CuSift.makeImageStruct(img);
            hPtr = libpointer('int32Ptr', int32(-1));
            calllib(obj.LibName, 'ExtractSiftFromImage', imgS, hPtr, extractOpts);
            obj.checkError('extractSift');
            handle = hPtr.Value;
        end

        % ── Match ────────────────────────────────────────────────────
        function matchSiftData(obj, handle1, handle2)
            %MATCHSIFTDATA  Match SIFT features between two handles.
            calllib(obj.LibName, 'MatchSiftData', int32(handle1), int32(handle2));
            obj.checkError('matchSiftData');
        end

        % ── Find homography ─────────────────────────────────────────
        function [H, numMatches] = findHomography(obj, handle, homogOpts)
            %FINDHOMOGRAPHY  Compute homography from matched features.
            %
            %   [H, numMatches] = obj.findHomography(handle, homogOpts)
            %
            %   H          - 3x3 double homography matrix.
            %   numMatches - number of inliers.
            hBuf = libpointer('singlePtr', single(zeros(1,9)));
            nPtr = libpointer('int32Ptr', int32(0));
            calllib(obj.LibName, 'FindHomography', int32(handle), hBuf, nPtr, homogOpts);
            obj.checkError('findHomography');
            H = double(reshape(hBuf.Value, [3 3]).');
            numMatches = double(nPtr.Value);
        end

        % ── Warp images ─────────────────────────────────────────────
        function [w1, w2] = warpImages(obj, img1, img2, H, useGPU)
            %WARPIMAGES  Warp two images using a homography.
            %
            %   [w1, w2] = obj.warpImages(img1, img2, H)
            %   [w1, w2] = obj.warpImages(img1, img2, H, true)
            %
            %   Returns H x W single matrices.
            if nargin < 5, useGPU = true; end
            imgS1 = CuSift.makeImageStruct(img1);
            imgS2 = CuSift.makeImageStruct(img2);
            hFlat = single(H.');     % row-major
            hFlat = hFlat(:);
            hPtr  = libpointer('singlePtr', hFlat);

            out1.host_img_ = libpointer('singlePtr');
            out1.width_    = int32(0);
            out1.height_   = int32(0);
            out2 = out1;

            calllib(obj.LibName, 'WarpImages', imgS1, imgS2, hPtr, out1, out2, int32(useGPU ~= 0));
            obj.checkError('warpImages');

            w1 = CuSift.imageStructToMatrix(out1);
            w2 = CuSift.imageStructToMatrix(out2);

            calllib(obj.LibName, 'FreeImage', out1);
            calllib(obj.LibName, 'FreeImage', out2);
        end

        % ── Delete SiftData ─────────────────────────────────────────
        function deleteSiftData(obj, handle)
            %DELETESIFTDATA  Free a SiftData handle.
            calllib(obj.LibName, 'DeleteSiftData', int32(handle));
            obj.checkError('deleteSiftData');
        end

        % ── Save to JSON ────────────────────────────────────────────
        function saveSiftData(obj, filename, handle)
            %SAVESIFTDATA  Save SIFT features to a JSON file.
            calllib(obj.LibName, 'SaveSiftData', filename, int32(handle));
            obj.checkError('saveSiftData');
        end

        % ── Accessor: number of points ──────────────────────────────
        function n = getNumPoints(obj, handle)
            %GETNUMPOINTS  Return the number of keypoints in a handle.
            n = calllib(obj.LibName, 'CusiftGetNumPoints', int32(handle));
            obj.checkError('getNumPoints');
        end

        % ── Accessor: single point ──────────────────────────────────
        function pt = getSiftPoint(obj, handle, index)
            %GETSIFTPOINT  Retrieve one SiftPoint struct (0-based index).
            pt.xpos        = single(0);
            pt.ypos        = single(0);
            pt.scale       = single(0);
            pt.sharpness   = single(0);
            pt.edgeness    = single(0);
            pt.orientation = single(0);
            pt.score       = single(0);
            pt.ambiguity   = single(0);
            pt.match       = int32(0);
            pt.match_xpos  = single(0);
            pt.match_ypos  = single(0);
            pt.match_error = single(0);
            pt.subsampling = single(0);
            pt.empty       = single(zeros(1,3));
            pt.data        = single(zeros(1,128));
            calllib(obj.LibName, 'CusiftGetSiftPoint', int32(handle), int32(index), pt);
            obj.checkError('getSiftPoint');
        end

        % ── Accessor: all points as a struct array ──────────────────
        function pts = getAllPoints(obj, handle)
            %GETALLPOINTS  Retrieve all keypoints as an N-element struct array.
            n = obj.getNumPoints(handle);
            if n == 0
                pts = struct([]);
                return;
            end
            pts(n) = struct('xpos',0,'ypos',0,'scale',0,'sharpness',0, ...
                            'edgeness',0,'orientation',0,'score',0, ...
                            'ambiguity',0,'match',0,'match_xpos',0, ...
                            'match_ypos',0,'match_error',0,'subsampling',0, ...
                            'descriptor',zeros(1,128,'single'));
            for i = 1:n
                sp = obj.getSiftPoint(handle, i-1);
                pts(i).xpos        = sp.xpos;
                pts(i).ypos        = sp.ypos;
                pts(i).scale       = sp.scale;
                pts(i).sharpness   = sp.sharpness;
                pts(i).edgeness    = sp.edgeness;
                pts(i).orientation = sp.orientation;
                pts(i).score       = sp.score;
                pts(i).ambiguity   = sp.ambiguity;
                pts(i).match       = sp.match;
                pts(i).match_xpos  = sp.match_xpos;
                pts(i).match_ypos  = sp.match_ypos;
                pts(i).match_error = sp.match_error;
                pts(i).subsampling = sp.subsampling;
                pts(i).descriptor  = sp.data;
            end
        end

        % ---------------------------------
        function [h1, h2] = extractAndMatchSift(obj, img1, img2, extractOpts)
            %EXTRACTANDMATCHSIFT  Extract and match in one call.
            %
            %   [h1, h2] = obj.extractAndMatchSift(img1, img2, extractOpts)
            imgS1 = CuSift.makeImageStruct(img1);
            imgS2 = CuSift.makeImageStruct(img2);
            h1Ptr = libpointer('int32Ptr', int32(-1));
            h2Ptr = libpointer('int32Ptr', int32(-1));
            calllib(obj.LibName, 'ExtractAndMatchSift', imgS1, imgS2, h1Ptr, h2Ptr, extractOpts);
            obj.checkError('extractAndMatchSift');
            h1 = h1Ptr.Value;
            h2 = h2Ptr.Value;
        end

        % ---------------------------------
        function [h1, h2, H, numMatches] = extractAndMatchAndFindHomography(obj, img1, img2, extractOpts, homogOpts)
            %EXTRACTANDMATCHANDFINDHOMOGRAPHY  Full pipeline minus warping.
            imgS1 = CuSift.makeImageStruct(img1);
            imgS2 = CuSift.makeImageStruct(img2);
            h1Ptr = libpointer('int32Ptr', int32(-1));
            h2Ptr = libpointer('int32Ptr', int32(-1));
            hBuf  = libpointer('singlePtr', single(zeros(1,9)));
            nPtr  = libpointer('int32Ptr', int32(0));
            calllib(obj.LibName, 'ExtractAndMatchAndFindHomography', ...
                    imgS1, imgS2, h1Ptr, h2Ptr, hBuf, nPtr, extractOpts, homogOpts);
            obj.checkError('extractAndMatchAndFindHomography');
            h1 = h1Ptr.Value;
            h2 = h2Ptr.Value;
            H  = double(reshape(hBuf.Value, [3 3]).');
            numMatches = double(nPtr.Value);
        end

        % ---------------------------------
        function [h1, h2, H, numMatches] = extractAndMatchAndFindHomographyMulti( ...
                obj, img1, img2, extractOpts, homogOpts, numAttempts, goal)
            %EXTRACTANDMATCHANDFINDHOMOGRAPHYMULTI  Multi-attempt homography.
            %
            %   goal: 0 = max inliers (default), 1 = min eye diff.
            if nargin < 6, numAttempts = 5;  end
            if nargin < 7, goal        = 0;  end
            imgS1 = CuSift.makeImageStruct(img1);
            imgS2 = CuSift.makeImageStruct(img2);
            h1Ptr = libpointer('int32Ptr', int32(-1));
            h2Ptr = libpointer('int32Ptr', int32(-1));
            hBuf  = libpointer('singlePtr', single(zeros(1,9)));
            nPtr  = libpointer('int32Ptr', int32(0));
            calllib(obj.LibName, 'ExtractAndMatchAndFindHomography_Multi', ...
                    imgS1, imgS2, h1Ptr, h2Ptr, hBuf, nPtr, ...
                    extractOpts, homogOpts, int32(numAttempts), int32(goal));
            obj.checkError('extractAndMatchAndFindHomographyMulti');
            h1 = h1Ptr.Value;
            h2 = h2Ptr.Value;
            H  = double(reshape(hBuf.Value, [3 3]).');
            numMatches = double(nPtr.Value);
        end

        % ---------------------------------
        function [h1, h2, H, numMatches, w1, w2] = extractMatchHomographyWarp( ...
                obj, img1, img2, extractOpts, homogOpts)
            %EXTRACTMATCHHOMOGRAPHYWARP  Full pipeline with CPU/GPU warp.
            imgS1 = CuSift.makeImageStruct(img1);
            imgS2 = CuSift.makeImageStruct(img2);
            h1Ptr = libpointer('int32Ptr', int32(-1));
            h2Ptr = libpointer('int32Ptr', int32(-1));
            hBuf  = libpointer('singlePtr', single(zeros(1,9)));
            nPtr  = libpointer('int32Ptr', int32(0));

            out1.host_img_ = libpointer('singlePtr');
            out1.width_    = int32(0);
            out1.height_   = int32(0);
            out2 = out1;

            calllib(obj.LibName, 'ExtractAndMatchAndFindHomographyAndWarp', ...
                    imgS1, imgS2, h1Ptr, h2Ptr, hBuf, nPtr, ...
                    extractOpts, homogOpts, out1, out2);
            obj.checkError('extractMatchHomographyWarp');

            h1 = h1Ptr.Value;
            h2 = h2Ptr.Value;
            H  = double(reshape(hBuf.Value, [3 3]).');
            numMatches = double(nPtr.Value);
            w1 = CuSift.imageStructToMatrix(out1);
            w2 = CuSift.imageStructToMatrix(out2);

            calllib(obj.LibName, 'FreeImage', out1);
            calllib(obj.LibName, 'FreeImage', out2);
        end

        % ---------------------------------
        function [h1, h2, H, numMatches, w1, w2] = extractMatchHomographyWarpMulti( ...
                obj, img1, img2, extractOpts, homogOpts, numAttempts, goal)
            %EXTRACTMATCHHOMOGRAPHYWARPMULTI  Multi-attempt full pipeline.
            if nargin < 6, numAttempts = 5;  end
            if nargin < 7, goal        = 0;  end
            imgS1 = CuSift.makeImageStruct(img1);
            imgS2 = CuSift.makeImageStruct(img2);
            h1Ptr = libpointer('int32Ptr', int32(-1));
            h2Ptr = libpointer('int32Ptr', int32(-1));
            hBuf  = libpointer('singlePtr', single(zeros(1,9)));
            nPtr  = libpointer('int32Ptr', int32(0));

            out1.host_img_ = libpointer('singlePtr');
            out1.width_    = int32(0);
            out1.height_   = int32(0);
            out2 = out1;

            calllib(obj.LibName, 'ExtractAndMatchAndFindHomography_Multi_AndWarp', ...
                    imgS1, imgS2, h1Ptr, h2Ptr, hBuf, nPtr, ...
                    extractOpts, homogOpts, out1, out2, ...
                    int32(numAttempts), int32(goal));
            obj.checkError('extractMatchHomographyWarpMulti');

            h1 = h1Ptr.Value;
            h2 = h2Ptr.Value;
            H  = double(reshape(hBuf.Value, [3 3]).');
            numMatches = double(nPtr.Value);
            w1 = CuSift.imageStructToMatrix(out1);
            w2 = CuSift.imageStructToMatrix(out2);

            calllib(obj.LibName, 'FreeImage', out1);
            calllib(obj.LibName, 'FreeImage', out2);
        end

        % ── VRAM estimation ─────────────────────────────────────────
        function bytes = estimateVramExtractSift(obj, width, height, extractOpts)
            %ESTIMATEVRAMEXTRACTSIFT  Estimate peak VRAM for extraction.
            bytes = calllib(obj.LibName, 'EstimateVramExtractSift', ...
                            int32(width), int32(height), extractOpts);
            obj.checkError('estimateVramExtractSift');
        end

        function bytes = estimateVramMatchSift(obj, maxKp1, maxKp2)
            bytes = calllib(obj.LibName, 'EstimateVramMatchSift', ...
                            int32(maxKp1), int32(maxKp2));
            obj.checkError('estimateVramMatchSift');
        end

        function bytes = estimateVramFindHomography(obj, maxKp, homogOpts)
            bytes = calllib(obj.LibName, 'EstimateVramFindHomography', ...
                            int32(maxKp), homogOpts);
            obj.checkError('estimateVramFindHomography');
        end

        function bytes = estimateVramWarpImages(obj, w1, h1, w2, h2)
            bytes = calllib(obj.LibName, 'EstimateVramWarpImages', ...
                            int32(w1), int32(h1), int32(w2), int32(h2));
            obj.checkError('estimateVramWarpImages');
        end

        function bytes = estimateVramFullPipeline(obj, w1, h1, w2, h2, extractOpts, homogOpts)
            bytes = calllib(obj.LibName, 'EstimateVramFullPipeline', ...
                            int32(w1), int32(h1), int32(w2), int32(h2), ...
                            extractOpts, homogOpts);
            obj.checkError('estimateVramFullPipeline');
        end
    end

    % ---------------------------------
    %  Private image conversion helper
    % ---------------------------------
    methods (Access = private, Static)
        function mat = imageStructToMatrix(imgStruct)
            %IMAGESTRUCTTOMATRIX  Convert a library-returned Image_t to an
            %   H x W single matrix.
            w = double(imgStruct.width_);
            h = double(imgStruct.height_);
            if w == 0 || h == 0
                mat = single([]);
                return;
            end
            % The library stores row-major data; read w*h floats, reshape,
            % then transpose back to MATLAB column-major.
            imgStruct.host_img_.setdatatype('singlePtr', w * h);
            raw = imgStruct.host_img_.Value;   % 1 x (w*h) single
            mat = reshape(raw, [w, h]).';       % H x W
        end
    end
end
