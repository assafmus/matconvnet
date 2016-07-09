classdef MapGenerator < dagnn.Layer
    properties
        opts = {}
        
        forceValidPositive = true;
        posOverlapThreshold = 0.5;
        negOverlapThreshold = 0.2;
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
            end
            
            [xs,ys] = meshgrid(1:size(inputs{1},2),1:size(inputs{1},1));
            xs = (xs-1)*obj.winStride(2)+1-obj.winPad(2);
            ys = (ys-1)*obj.winStride(1)+1-obj.winPad(1);
            grid_rects = [xs(:) ys(:) repmat(obj.winSize([2 1]), numel(xs),1)];            
            
            sz = obj.getOutputSizes(cellfun(@size, inputs, 'uniformoutput',false));
            outputs = {gpuArray(zeros(sz{1},'single')) gpuArray(nan(sz{2},'single'))};
            
            for i = 1 : size(inputs{1},4)
                all_rects = inputs{2}{i}(:,1:4);
                valid_flag = ~~inputs{2}{i}(:,5);
                pos_rects = all_rects(valid_flag,:);
                
                Oall = rectOverlap(grid_rects, all_rects);
                Oneg = max(Oall,[],2);
                neg = Oneg < obj.negOverlapThreshold;

                if obj.forceValidPositive
                    Opos = rectOverlap(grid_rects, pos_rects);
                else
                    Opos = Oall;
                end
                [Opos,Ipos] = max(Opos,[],2);
                pos = Opos > obj.posOverlapThreshold;
                
                tmp = outputs{1}(:,:,:,i);
                tmp(pos) = 2;
                tmp(neg) = 1;
                outputs{1}(:,:,:,i) = tmp;
                                        
                tmp = outputs{2}(:,:,:,i);
                tmp = permute(tmp,[3 1 2]);
                tmp(1,pos) = (pos_rects(Ipos(pos),1) - grid_rects(pos,1)) ./ grid_rects(pos,3);
                tmp(2,pos) = (pos_rects(Ipos(pos),2) - grid_rects(pos,2)) ./ grid_rects(pos,4);
                tmp(3,pos) = log2(pos_rects(Ipos(pos),3) ./ grid_rects(pos,3));
                tmp(4,pos) = log2(pos_rects(Ipos(pos),4) ./ grid_rects(pos,4));
                tmp = permute(tmp,[2 3 1]);
                outputs{2}(:,:,:,i) = tmp;
                
%                 input = obj.net.getVar('input').value;
%                 imshow(input(:,:,:,i))
%                 plotRect(inputs{2}{i}(:,1:4),'y');
%                 plotRect(inputs{2}{i}(~~inputs{2}{i}(:,5),1:4),'g');
                
                
                % plotRect(grid_rects(neg,:),'r')
                % plotRect(grid_rects(~neg&~pos,:),'y')
                % plotRect(grid_rects(pos,:),'g')
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
            outputSizes{2} = [inputSizes{1}(1) inputSizes{1}(2) 4 inputSizes{1}(4)] ;
        end
        
        function rfs = getReceptiveFields(obj)
            % the receptive field depends on the dimension of the variables
            % which is not known until the network is run
            rfs.size = [NaN NaN] ;
            rfs.stride = [NaN NaN] ;
            rfs.offset = [NaN NaN] ;
            rfs = repmat(rfs,2,2);
        end
        
        function obj = MapGenerator(varargin)
            obj.load(varargin) ;
        end
    end
end
