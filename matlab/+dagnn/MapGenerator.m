classdef MapGenerator < dagnn.Layer
    properties
        opts = {}
    end
    
    properties (Transient)
        winSize = [];
        winStride = [];
        winPad = [];
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            if isempty(obj.winSize) || isempty(obj.winStride) || isempty(obj.winPad)
                rf = obj.net.getVarReceptiveFields('input');
                pred = obj.net.layers(obj.layerIndex).inputIndexes(1);
                rf = rf(pred);
                obj.winSize = rf.size;
                obj.winStride = rf.stride;
                obj.winPad = (obj.winSize+1)/2 - rf.offset;
                assert(~any(obj.winPad))
            end
            
            [xs,ys] = meshgrid(1:size(inputs{1},2),1:size(inputs{1},1));
            xs = (xs-1)*obj.winStride(2)+1;
            ys = (ys-1)*obj.winStride(1)+1;
            grid_rects = [xs(:) ys(:) repmat(obj.winSize([2 1]), numel(xs),1)];
            
            
            sz = obj.getOutputSizes(cellfun(@size, inputs, 'uniformoutput',false));
            outputs = {gpuArray(zeros(sz{1},'single'))};
            
            for i = 1 : size(inputs{1},4)
                O = rectOverlap(grid_rects, inputs{2}{i});
                O = max(O,[],2);
                pos = O > .5;
                neg = O < .2;
                tmp = outputs{1}(:,:,:,i);
                tmp(pos) = 2;
                tmp(neg) = 1;
                outputs{1}(:,:,:,i) = tmp;
                
%                 input = obj.net.getVar('input').value;
%                 imshow(input(:,:,:,i))
%                 plotRect(grid_rects(neg,:),'r')
%                 plotRect(grid_rects(~neg&~pos,:),'y')
%                 plotRect(grid_rects(pos,:),'g')
            end
            
            
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            error('backward');   
        end
        
        function reset(obj)
            obj.winSize = [];
            obj.winStride = [];
            obj.winPad = [];
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
            inputSizes{1} = [inputSizes{1} 1 1 1 1];
            inputSizes{1} = inputSizes{1}(1:4);
            outputSizes{1} = [inputSizes{1}(1) inputSizes{1}(2) 1 inputSizes{1}(4)] ;
        end
        
        function rfs = getReceptiveFields(obj)
            % the receptive field depends on the dimension of the variables
            % which is not known until the network is run
            rfs(1,1).size = [NaN NaN] ;
            rfs(1,1).stride = [NaN NaN] ;
            rfs(1,1).offset = [NaN NaN] ;
            rfs(2,1) = rfs(1,1) ;
        end
        
        function obj = MapGenerator(varargin)
            obj.load(varargin) ;
        end
    end
end
