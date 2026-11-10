classdef CuSiftKeypoints < handle
    %CUSIFTKEYPOINTS A keypoint set living inside the cusift_matlab library.
    %   Returned by CuSift.extract and the pipeline methods.  The set stays on
    %   the GPU and the host inside the library; this object holds its handle
    %   and fetches a copy of the data lazily.  Deleting the object (or
    %   calling free) releases the set.
    %
    %   kp.Count                 number of keypoints
    %   kp.data()                144 x Count single, one keypoint per column
    %                            (row layout documented in cusift_matlab.h)
    %   kp.xy()                  Count x 2 double, 0-based pixel coordinates
    %   kp.scale(), kp.orientation(), kp.score(), kp.ambiguity()
    %   kp.match()               Count x 1, 1-based index into the other set
    %                            after CuSift.match, 0 where unmatched
    %   kp.matchXY()             Count x 2, coordinates of the matched keypoint
    %   kp.matchError()          reprojection error after estimateHomography
    %   kp.descriptors()         128 x Count single
    %   kp.saveJson(path)
    %   kp.free()

    properties (SetAccess = private)
        Handle (1,1) uint64 = uint64(0)
        Count (1,1) double = 0
        Capacity (1,1) double = 0
    end

    properties (Access = private)
        Data = []   % cached copy, dropped whenever the library changes the set
    end

    methods
        function obj = CuSiftKeypoints(handle)
            %CUSIFTKEYPOINTS Wrap a handle from the library.  Internal.
            obj.Handle = uint64(handle);
            obj.refresh();
        end

        function refresh(obj)
            %REFRESH Re-read count and capacity and drop the cached copy.
            if obj.Handle == 0, return; end
            count = libpointer('int32Ptr', int32(0));
            capacity = libpointer('int32Ptr', int32(0));
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_keypoints_info', obj.Handle, count, capacity));
            obj.Count = double(count.Value);
            obj.Capacity = double(capacity.Value);
            obj.Data = [];
        end

        function d = data(obj)
            %DATA The 144 x Count single matrix of the set (cached).
            if isempty(obj.Data)
                if obj.Count == 0
                    obj.Data = zeros(144, 0, 'single');
                else
                    buf = libpointer('singlePtr', zeros(144, obj.Count, 'single'));
                    CuSift.check(calllib(CuSift.Lib, 'cusiftml_keypoints_copy', obj.Handle, buf, int32(obj.Count)));
                    obj.Data = reshape(buf.Value, 144, obj.Count);
                end
            end
            d = obj.Data;
        end

        function v = xy(obj);          d = obj.data(); v = double(d(1:2, :)).'; end
        function v = scale(obj);       d = obj.data(); v = double(d(3, :)).'; end
        function v = sharpness(obj);   d = obj.data(); v = double(d(4, :)).'; end
        function v = edgeness(obj);    d = obj.data(); v = double(d(5, :)).'; end
        function v = orientation(obj); d = obj.data(); v = double(d(6, :)).'; end
        function v = score(obj);       d = obj.data(); v = double(d(7, :)).'; end
        function v = ambiguity(obj);   d = obj.data(); v = double(d(8, :)).'; end
        function v = matchXY(obj);     d = obj.data(); v = double(d(10:11, :)).'; end
        function v = matchError(obj);  d = obj.data(); v = double(d(12, :)).'; end
        function v = subsampling(obj); d = obj.data(); v = double(d(13, :)).'; end
        function v = descriptors(obj); d = obj.data(); v = d(17:144, :); end

        function v = match(obj)
            %MATCH 1-based index of each keypoint's match in the other set; 0 = none.
            d = obj.data();
            v = double(typecast(d(9, :), 'int32')).' + 1;   % C gives 0-based or -1
        end

        function saveJson(obj, path)
            %SAVEJSON Write the set, descriptors included, to a JSON file.
            CuSift.check(calllib(CuSift.Lib, 'cusiftml_keypoints_save_json', obj.Handle, char(path)));
        end

        function free(obj)
            %FREE Release the set inside the library.  Safe to call twice.
            if obj.Handle ~= 0 && libisloaded(CuSift.Lib)
                calllib(CuSift.Lib, 'cusiftml_keypoints_free', obj.Handle);
            end
            obj.Handle = uint64(0);
            obj.Count = 0;
            obj.Data = [];
        end

        function delete(obj)
            obj.free();
        end

        function invalidate(obj)
            %INVALIDATE Drop the cached copy (after match / estimateHomography).  Internal.
            obj.Data = [];
        end
    end
end
