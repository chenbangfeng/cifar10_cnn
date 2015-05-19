require 'torch'
require 'nn'

function load_model(filename)
	return torch.load(filename)
end
