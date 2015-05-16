require 'torch'
require 'nn'
--model = torch.load('results_model2_YUV/model.net')
--print(model:get(4).weight)

function load_model(filename)
	return torch.load(filename)
end

function visualize_kernels(filename, layer, filtsize)
	model = load_model(filename)
	kernels = model:get(layer).weight
	kernels_num = kernels:size(1)
	channels_num = kernels:size(2)/(filtsize*filtsize)
	kernels = kernels:resize(kernels_num, channels_num, filtsize, filtsize)
	if itorch then
		if channels_num == 3 then
			itorch.image(kernels)
		else 
			for i = 1, channels_num do
				itorch.image(kernels[{ {}, i }])
			end
		end
	else
		print("For visualization, run this script in an itorch notebook")
	end
end

visualize_kernels('results_model2_YUV/model.net', 1, 3)
visualize_kernels('results_model2_YUV/model.net', 4, 3)
model = load_model('results_model2_YUV/model.net')
