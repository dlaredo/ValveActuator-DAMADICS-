function blkStruct = slblocks
%SLBLOCKS Defines the block library for a specific Toolbox or Blockset.

%   Copyright (c) 1990-1998 by The MathWorks, Inc. All Rights Reserved.
%   $Revision: 1.6 $

% Name of the subsystem which will show up in the Simulink Blocksets
% and Toolboxes subsystem.
% Example:  blkStruct.Name = 'DSP Blockset';
blkStruct.Name = ['DAMADICS' sprintf('\n') 'Actuator' sprintf('\n') 'Benchmark' sprintf('\n') 'Library'];

% The function that will be called when the user double-clicks on
% this icon.
% Example:  blkStruct.OpenFcn = 'dsplib';
blkStruct.OpenFcn = 'DABLib';

% The argument to be set as the Mask Display for the subsystem.  You
% may comment this line out if no specific mask is desired.
% Example:  blkStruct.MaskDisplay = 'plot([0:2*pi],sin([0:2*pi]));';
% No display for now.
blkStruct.MaskDisplay = 'plot( [0.2 0.8], [0.3 0.3] ), plot( [0.65 0.65 0.35 0.35 0.65], [0.1 0.5 0.1 0.5 0.1] ), plot( [0.5 0.5], [0.3 0.6]), plot( [0.35 0.65 0.60 0.4 0.35], [0.6 0.6 0.7 0.7 0.6] )';

% End of slblocks


