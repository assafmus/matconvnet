function obj = fromSimpleNN(net, varargin)
% FROMSIMPLENN  Initialize a DagNN object from a SimpleNN network
%   FROMSIMPLENN(NET) initializes the DagNN object from the
%   specified CNN using the SimpleNN format.
%
%   SimpleNN objects are linear chains of computational layers. These
%   layers echange information through variables and parameters that
%   are not explicitly named.Hence, FROMSIMPLENN() uses a number of
%   rules to assign such names automatically:
%
%   * From the input to the output of the CNN, variables are called
%     `x0` (input of the first layer), `x1`, `x2`, .... In this
%     manner `xi` is the outut of the i-th layer.
%
%   * Any loss layer requires two inputs, the second being a label.
%     These are called `label` (for the first such layers), and then
%     `label2`, `label3`,... for any other similar layer.
%
%   Additinoally, the option `CanonicalNames` the function can change
%   the names of some variables to make them more convenient to
%   use. With this option turned on:
%
%   * The network input is called `input` instead of `x0`.
%
%   * The output of each SoftMax layer is called `prob` (or `prob2`,
%     ...).
%
%   * The output of each Loss layer is called `objective` (or `
%     objective2`, ...).
%
%   * The input of each SoftMax or Loss layer of type *softmax log
%     loss* is called `prediction` (or `prediction2`, ...). If a Loss
%     layer immediately follows a SoftMax layer, then the rule above
%     takes the precendence and the input name is not changed.
%
%   FROMSIMPLENN(___, 'OPT', VAL, ...) accepts the following options:
%
%   `CanonicalNames`:: false
%      If `true` use the rules above to assign more meaningful
%      names to some of the varibles.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

import dagnn.*
obj = DagNN() ;

% copy meta-information as is
obj.meta = net.meta ;
obj.addLayersFromSimpleNN(net, varargin{:});
