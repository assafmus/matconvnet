function renameParam(obj, oldName, newName, varargin)
%RENAMEPARAM Rename a parameter
%   RENAMEPARAM(OLDNAME, NEWNAME) changes the name of the parameter
%   OLDNAME into NEWNAME. NEWNAME should not be the name of an
%   existing parameter.

opts.quiet = false ;
opts = vl_argparse(opts, varargin) ;

% Find the parameter to rename
v = obj.getParamIndex(oldName) ;
if isnan(v)
  % There is no such a parameter, nothing to do
  if ~opts.quiet
    warning('There is no parameter ''%s''.', oldName) ;
  end
  return ;
end

% Check if newName is an existing parameter
newNameExists = any(strcmp(newName, {obj.params.name})) ;

% Replace oldName with newName in all the layers
for l = 1:numel(obj.layers)
  for f = {'params'}
     f = char(f) ;
     sel = find(strcmp(oldName, obj.layers(l).(f))) ;
     [obj.layers(l).(f){sel}] = deal(newName) ;
  end
end

% If newParameter is a parameter in the graph, then there is not
% anything else to do. obj.rebuild() will remove the slot
% in obj.params() for oldName as that parameter becomes unused.
%
% If, however, newparameter is not in the graph already, then
% the slot in obj.params() is preserved and only the parameter name
% is changed.

if ~newNameExists
  obj.params(v).name = newName ;
  % update parameter name hash otherwise rebuild() won't find this
  % parameter corectly
  obj.paramNames.(newName) = v ;
end

obj.rebuild() ;
