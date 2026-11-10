function lib = cusift_load(libdir, makePrototype)
%CUSIFT_LOAD Load the cusift_matlab shared library into MATLAB.
%   LIB = CUSIFT_LOAD() loads cusift_matlab.dll (Windows) or
%   libcusift_matlab.so (Linux), looking next to this file first and then in
%   the usual build and install directories of the source tree.  It returns
%   the alias used with calllib, 'cusift_matlab', and does nothing if the
%   library is already loaded.
%
%   LIB = CUSIFT_LOAD(DIR) loads it from DIR.
%
%   CUSIFT_LOAD(DIR, true) also writes a prototype file,
%   cusift_matlab_proto.m, next to this function.  loadlibrary parses
%   include/cusift_matlab.h with the C compiler selected by "mex -setup C"
%   (the free MinGW support package is enough); once the prototype file
%   exists, later loads use it and need no compiler at all.  Pass '' as DIR
%   to keep the automatic search.
%
%   The CuSift class calls this for you.  Before unloadlibrary, call
%   CuSift.freeAll (or cusiftml_free_all) so the GPU memory behind any live
%   handles is released; unloading alone does not do that.

lib = 'cusift_matlab';
if nargin < 1, libdir = ''; end
if nargin < 2, makePrototype = false; end

if libisloaded(lib)
    if ~makePrototype, return; end
    unloadlibrary(lib);
end

here = fileparts(mfilename('fullpath'));
if ispc
    fname = 'cusift_matlab.dll';
else
    fname = 'libcusift_matlab.so';
end

if isempty(libdir)
    candidates = { ...
        here, ...
        fullfile(here, '..', 'build', 'Release'), ...
        fullfile(here, '..', 'build'), ...
        fullfile(here, '..', 'install', 'bin'), ...
        fullfile(here, '..', 'install', 'lib'), ...
        fullfile(here, '..', '..', 'bin'), ...      % <prefix>/share/cusift/matlab -> <prefix>/bin
        fullfile(here, '..', '..', 'lib')};         % <prefix>/share/cusift/matlab -> <prefix>/lib
    libdir = '';
    for k = 1:numel(candidates)
        if exist(fullfile(candidates{k}, fname), 'file')
            libdir = candidates{k};
            break;
        end
    end
    if isempty(libdir)
        error('cusift:load', ['Cannot find %s. Build with -DCUSIFT_BUILD_MATLAB=ON ' ...
            '-DCUSIFT_STATIC_RUNTIME=ON, then pass its directory: cusift_load(dir).'], fname);
    end
end
libfile = fullfile(libdir, fname);
if ~exist(libfile, 'file')
    error('cusift:load', 'No %s in %s.', fname, libdir);
end

header = fullfile(here, '..', 'include', 'cusift_matlab.h');
if ~exist(header, 'file')
    header = fullfile(here, '..', '..', '..', 'include', 'cusift_matlab.h');  % installed layout
end
proto = fullfile(here, 'cusift_matlab_proto.m');

if makePrototype
    old = cd(here);
    restore = onCleanup(@() cd(old));
    loadlibrary(libfile, header, 'alias', lib, 'mfilename', 'cusift_matlab_proto');
    fprintf('Wrote %s\n', proto);
elseif exist(proto, 'file')
    loadlibrary(libfile, @cusift_matlab_proto, 'alias', lib);
else
    if ~exist(header, 'file')
        error('cusift:load', 'Cannot find cusift_matlab.h and no prototype file exists; run cusift_load(dir, true) once from a checkout.');
    end
    loadlibrary(libfile, header, 'alias', lib);
end
end
