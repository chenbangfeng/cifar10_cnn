require 'torch'
require 'image'
require 'nn'
--dofile 'load_model.lua'


function visualize_input(trainData, first, last, sep_channel)
	if itorch then
		samples_y = trainData.data[{ {first, last}, 1 }]
		samples_u = trainData.data[{ {first, last}, 2 }]
		samples_v = trainData.data[{ {first, last}, 3 }]
		samples = trainData.data[{ {first, last} }]
		if sep_channel then
			print('Images in first channel')
			itorch.image(samples_y)
			print('Images in second channel')
			itorch.image(samples_u)
			print('Images in third channel')
			itorch.image(samples_v)
		end
		itorch.image(samples)
	else
		print("For visualization, run this script in an itorch notebook")
	end
end

function visualize_kernels(model, layer, filtsize)
	-- model = load_model(filename)
	kernels = model:get(layer).weight
	kernels_num = kernels:size(1)
	channels_num = kernels:size(2)/(filtsize*filtsize)
	kernels = kernels:resize(kernels_num, channels_num, filtsize, filtsize)
	if itorch then
		-- proper 3 channels image
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

function visualize_saved_model(filename, layer, filtsize)
  model = load_model(filename)
  visualize_kernels(model, layer, filtsize)
end

