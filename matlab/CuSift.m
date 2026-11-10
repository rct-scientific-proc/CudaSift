classdef CuSift < handle
    %CUSIFT GPU SIFT extraction, matching, homography estimation and warping.
    %   A thin class over the cusift_matlab shared library (loaded with
    %   loadlibrary, no MEX).  Images are 2-D grayscale matrices of any numeric
    %   class with values in [0, 255]; they are converted to single and
    %   transposed for the library here, so pass them as they come from imread.
    %
    %   sift = CuSift();                                 % loads the library, selects the GPU
    %   kp1 = sift.extract(img1);                        % CuSiftKeypoints
    %   kp2 = sift.extract(img2);
    %   sift.match(kp1, kp2);                            % fills kp1's match rows
    %   [H, inliers] = sift.estimateHomography(kp1);    % 3x3, image 1 -> image 2
    %   [w1, w2] = sift.warp(img1, img2, H);             % common canvas, NaN where uncovered
    %
    %   [kp1, kp2, H, inliers] = sift.align(img1, img2); % the fused pipeline
    %   [kp1, kp2, H, inliers, w1, w2] = sift.alignAndWarp(img1, img2);
    %
    %   Options are plain structs from sift.extractOptions() and
    %   sift.homographyOptions(), with name/value overrides:
    %   opts = sift.homographyOptions('seed', 42, 'attempts', 3);
    %
    %   Every library failure raises a MATLAB error 'cusift:error' with the
    %   library's message.  See matlab/README.md for building and loading.

    properties (Constant)
        Lib = 'cusift_matlab';
    end

    methods
        function obj = CuSift(libdir)
            %CUSIFT Load the library (if needed) and select the CUDA device.
            if nargin < 1, libdir = ''; end
            cusift_load(libdir);
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_init'));
        end

        function v = version(~)
            %VERSION [major minor patch] of the loaded library.
            a = libpointer('int32Ptr', int32(0));
            b = libpointer('int32Ptr', int32(0));
            c = libpointer('int32Ptr', int32(0));
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_version', a, b, c));
            v = double([a.Value b.Value c.Value]);
        end

        % -- Options ------------------------------------------------------

        function opts = extractOptions(~, varargin)
            %EXTRACTOPTIONS Default extraction options, with name/value overrides.
            p = libstruct('cusiftml_extract_options');
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_default_extract_options', p));
            opts = CuSift.override(get(p), varargin{:});
        end

        function opts = homographyOptions(~, varargin)
            %HOMOGRAPHYOPTIONS Default homography options, with name/value overrides.
            %   model: 0 homography, 1 similarity.  goal: 0 most inliers,
            %   1 closest to identity (used when attempts > 1).
            p = libstruct('cusiftml_homography_options');
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_default_homography_options', p));
            opts = CuSift.override(get(p), varargin{:});
        end

        % -- Stages -------------------------------------------------------

        function kp = extract(obj, img, opts)
            %EXTRACT Detect SIFT keypoints and descriptors on the GPU.
            if nargin < 3 || isempty(opts), opts = obj.extractOptions(); end
            [pix, w, h] = CuSift.prepare(img);
            hp = libpointer('uint64Ptr', uint64(0));
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_extract', pix, w, h, opts, hp));
            kp = CuSiftKeypoints(hp.Value);
        end

        function match(~, query, train)
            %MATCH Match every keypoint of query against train; fills query's match rows.
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_match', query.Handle, train.Handle));
            query.invalidate();
        end

        function [H, inliers] = estimateHomography(obj, matched, opts)
            %ESTIMATEHOMOGRAPHY RANSAC plus refinement on a matched set.
            %   H is 3x3 with H(3,3) = 1, mapping image-1 points to image 2.
            if nargin < 3 || isempty(opts), opts = obj.homographyOptions(); end
            hp = libpointer('singlePtr', zeros(1, 9, 'single'));
            ip = libpointer('int32Ptr', int32(0));
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_estimate_homography', matched.Handle, opts, hp, ip));
            H = CuSift.toMatrix(hp.Value);
            inliers = double(ip.Value);
            matched.invalidate();
        end

        function canvas = computeCanvas(~, H, size1, size2)
            %COMPUTECANVAS The canvas a warp would produce, without warping.
            %   size1, size2: [height width] as size(img) gives them.
            %   canvas.origin [u v], canvas.size [width height],
            %   canvas.corners1 / corners2: 4x2 corner positions on the canvas.
            cp = libpointer('int32Ptr', zeros(1, 4, 'int32'));
            kp = libpointer('singlePtr', zeros(1, 16, 'single'));
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_compute_canvas', CuSift.toArray(H), ...
                size1(2), size1(1), size2(2), size2(1), cp, kp));
            c = double(cp.Value);
            k = reshape(double(kp.Value), 2, 8).';
            canvas = struct('origin', c(1:2), 'size', c(3:4), 'corners1', k(1:4, :), 'corners2', k(5:8, :));
        end

        function [w1, w2] = warp(~, img1, img2, H, useGpu)
            %WARP Warp both images onto the common canvas (NaN where uncovered).
            %   useGpu (default true): false runs the OpenMP host warp.
            if nargin < 5, useGpu = true; end
            [p1, w1_, h1_] = CuSift.prepare(img1);
            [p2, w2_, h2_] = CuSift.prepare(img2);
            a = libpointer('uint64Ptr', uint64(0));
            b = libpointer('uint64Ptr', uint64(0));
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_warp', p1, w1_, h1_, p2, w2_, h2_, ...
                CuSift.toArray(H), int32(logical(useGpu)), a, b));
            w1 = CuSift.takeImage(a.Value);
            w2 = CuSift.takeImage(b.Value);
        end

        % -- Pipelines ----------------------------------------------------

        function [kp1, kp2] = extractAndMatch(obj, img1, img2, opts)
            %EXTRACTANDMATCH Extract both sets and match kp1 against kp2.
            if nargin < 4 || isempty(opts), opts = obj.extractOptions(); end
            [p1, w1, h1] = CuSift.prepare(img1);
            [p2, w2, h2] = CuSift.prepare(img2);
            a = libpointer('uint64Ptr', uint64(0));
            b = libpointer('uint64Ptr', uint64(0));
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_extract_and_match', p1, w1, h1, p2, w2, h2, opts, a, b));
            kp1 = CuSiftKeypoints(a.Value);
            kp2 = CuSiftKeypoints(b.Value);
        end

        function [kp1, kp2, H, inliers] = align(obj, img1, img2, extractOpts, homographyOpts)
            %ALIGN Extract, match and estimate the homography in one call.
            if nargin < 4 || isempty(extractOpts), extractOpts = obj.extractOptions(); end
            if nargin < 5 || isempty(homographyOpts), homographyOpts = obj.homographyOptions(); end
            [p1, w1, h1] = CuSift.prepare(img1);
            [p2, w2, h2] = CuSift.prepare(img2);
            a = libpointer('uint64Ptr', uint64(0));
            b = libpointer('uint64Ptr', uint64(0));
            hp = libpointer('singlePtr', zeros(1, 9, 'single'));
            ip = libpointer('int32Ptr', int32(0));
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_align', p1, w1, h1, p2, w2, h2, ...
                extractOpts, homographyOpts, a, b, hp, ip));
            kp1 = CuSiftKeypoints(a.Value);
            kp2 = CuSiftKeypoints(b.Value);
            H = CuSift.toMatrix(hp.Value);
            inliers = double(ip.Value);
        end

        function [kp1, kp2, H, inliers, w1, w2] = alignAndWarp(obj, img1, img2, extractOpts, homographyOpts)
            %ALIGNANDWARP align followed by the GPU warp, reusing the uploaded images.
            if nargin < 4 || isempty(extractOpts), extractOpts = obj.extractOptions(); end
            if nargin < 5 || isempty(homographyOpts), homographyOpts = obj.homographyOptions(); end
            [p1, w1_, h1_] = CuSift.prepare(img1);
            [p2, w2_, h2_] = CuSift.prepare(img2);
            a = libpointer('uint64Ptr', uint64(0));
            b = libpointer('uint64Ptr', uint64(0));
            hp = libpointer('singlePtr', zeros(1, 9, 'single'));
            ip = libpointer('int32Ptr', int32(0));
            c = libpointer('uint64Ptr', uint64(0));
            d = libpointer('uint64Ptr', uint64(0));
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_align_and_warp', p1, w1_, h1_, p2, w2_, h2_, ...
                extractOpts, homographyOpts, a, b, hp, ip, c, d));
            kp1 = CuSiftKeypoints(a.Value);
            kp2 = CuSiftKeypoints(b.Value);
            H = CuSift.toMatrix(hp.Value);
            inliers = double(ip.Value);
            w1 = CuSift.takeImage(c.Value);
            w2 = CuSift.takeImage(d.Value);
        end

        % -- Device memory estimates (bytes) -----------------------------

        function b = memoryExtract(obj, imageSize, opts)
            if nargin < 3 || isempty(opts), opts = obj.extractOptions(); end
            b = CuSift.bytes(calllib(CuSift.Lib, 'cusiftml_memory_extract', imageSize(2), imageSize(1), opts));
        end
        function b = memoryMatch(~, capacity1, capacity2)
            b = CuSift.bytes(calllib(CuSift.Lib, 'cusiftml_memory_match', capacity1, capacity2));
        end
        function b = memoryHomography(obj, capacity, opts)
            if nargin < 3 || isempty(opts), opts = obj.homographyOptions(); end
            b = CuSift.bytes(calllib(CuSift.Lib, 'cusiftml_memory_homography', capacity, opts));
        end
        function b = memoryWarp(~, size1, size2)
            b = CuSift.bytes(calllib(CuSift.Lib, 'cusiftml_memory_warp', size1(2), size1(1), size2(2), size2(1)));
        end
        function b = memoryPipeline(obj, size1, size2, extractOpts, homographyOpts)
            if nargin < 4 || isempty(extractOpts), extractOpts = obj.extractOptions(); end
            if nargin < 5 || isempty(homographyOpts), homographyOpts = obj.homographyOptions(); end
            b = CuSift.bytes(calllib(CuSift.Lib, 'cusiftml_memory_pipeline', ...
                size1(2), size1(1), size2(2), size2(1), extractOpts, homographyOpts));
        end

        % -- Errors and device --------------------------------------------

        function setErrorLog(~, path, maxBytes)
            %SETERRORLOG Append every library error to a file ('' = <temp>/cusift-errors.log).
            if nargin < 2, path = ''; end
            if nargin < 3, maxBytes = 0; end
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_set_error_log', char(path), uint64(maxBytes)));
        end

        function clearErrorLog(~)
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_clear_error_log'));
        end

        function n = errorLogDropped(~)
            n = double(calllib(CuSift.Lib, 'cusiftml_error_log_dropped'));
        end

        function resetDevice(~)
            %RESETDEVICE Recover from a sticky CUDA error.  Frees every keypoint set and image first.
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_device_reset'));
        end
    end

    methods (Static)
        function freeAll()
            %FREEALL Release every keypoint set and warped image inside the library.
            %   Call before unloadlibrary('cusift_matlab').
            if libisloaded(CuSift.Lib)
                calllib(CuSift.Lib, 'cusiftml_free_all');
            end
        end

        function unload()
            %UNLOAD freeAll followed by unloadlibrary.
            CuSift.freeAll();
            if libisloaded(CuSift.Lib)
                unloadlibrary(CuSift.Lib);
            end
        end

        function msg = lastError()
            %LASTERROR Message of the most recent failed library call ('' if it succeeded).
            try
                [~, msg] = calllib(CuSift.Lib, 'cusiftml_last_error', blanks(2048), int32(2048));
                msg = strtrim(char(msg));
            catch
                msg = '';
            end
        end

        function check(status)
            %CHECK Raise 'cusift:error' with the library's message when status ~= 0.
            if status ~= 0
                msg = CuSift.lastError();
                if isempty(msg), msg = sprintf('library call failed with status %d', status); end
                error('cusift:error', '%s', msg);
            end
        end

        function [pix, width, height] = prepare(img)
            %PREPARE Convert a grayscale image to the row-major single array the library wants.
            validateattributes(img, {'numeric', 'logical'}, {'2d', 'nonempty'}, 'CuSift', 'img');
            height = size(img, 1);
            width = size(img, 2);
            pix = single(img).';     % column-major W x H is row-major H x W
        end

        function img = takeImage(handle)
            %TAKEIMAGE Copy a warped image out of the library and release it.  Internal.
            handle = uint64(handle);
            wp = libpointer('int32Ptr', int32(0));
            hp = libpointer('int32Ptr', int32(0));
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_image_info', handle, wp, hp));
            w = double(wp.Value);
            h = double(hp.Value);
            buf = libpointer('singlePtr', zeros(w, h, 'single'));
            status = calllib(CuSift.Lib, 'cusiftml_image_copy', handle, buf, int32(w), int32(h));
            calllib(CuSift.Lib, 'cusiftml_image_free', handle);
            CuSift.check(status);
            img = reshape(buf.Value, w, h).';
        end

        function H = toMatrix(h9)
            %TOMATRIX 9 row-major floats -> 3x3 double.
            H = reshape(double(h9), 3, 3).';
        end

        function h9 = toArray(H)
            %TOARRAY 3x3 -> 9 row-major singles.
            validateattributes(H, {'numeric'}, {'size', [3 3]}, 'CuSift', 'H');
            h9 = single(reshape(H.', 1, 9));
        end

        function b = bytes(v)
            if v == 0
                msg = CuSift.lastError();
                if ~isempty(msg), error('cusift:error', '%s', msg); end
            end
            b = double(v);
        end

        function s = override(s, varargin)
            %OVERRIDE Apply name/value pairs to an options struct, checking the names.
            if mod(numel(varargin), 2) ~= 0
                error('cusift:options', 'options must be name/value pairs');
            end
            for k = 1:2:numel(varargin)
                name = varargin{k};
                if ~isfield(s, name)
                    error('cusift:options', 'unknown option "%s"; valid: %s', name, strjoin(fieldnames(s).', ', '));
                end
                s.(name) = varargin{k + 1};
            end
        end
    end
end
