function reset(obj)
%RESET Reset the DagNN
%   RESET(obj) resets the DagNN obj. The function clears any intermediate value stored in the DagNN
%   object, including parameter gradients. It also calls the reset
%   function of every layer.

[obj.vars.value] = deal([]) ;
[obj.vars.der] = deal([]) ;
if ~isempty(obj.params), [obj.params.der] = deal([]) ; end;
for l = 1:numel(obj.layers)
  obj.layers(l).block.reset() ;
end
