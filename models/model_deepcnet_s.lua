require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

function init_model()
	print '==> define parameters'

	-- 10-class problem
	noutputs = 10

	-- input dimensions
	nfeats = 3
	width = 48
	height = 48
	ninputs = nfeats*width*height

	-- hidden units, filter sizes (for ConvNet only):
	nstates = {64, 64, 128, 256}
	filtsize = 3 --could be 5
	poolsize = 2
	normkernel = image.gaussian1D(7) -- not necessery

	----------------------------------------------------------------------
	print '==> construct model'

	--cnet(4, 100) on 48x48 data
	--100C3-MP2-200C2-MP2-300C2-MP2-400C2-MP2-500C2-softmax
	model = nn.Sequential()

	model:add(nn.SpatialConvolutionMM(3, 50, 3, 3))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

	model:add(nn.SpatialConvolutionMM(50, 100, 2, 2))
	model:add(nn.ReLU())
	model:add(nn.Dropout(0.1))
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))


	model:add(nn.SpatialConvolutionMM(100, 150, 2, 2))
	model:add(nn.ReLU())
	model:add(nn.Dropout(0.25))
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	
	model:add(nn.SpatialConvolutionMM(150, 200, 2, 2))
	model:add(nn.ReLU())
	model:add(nn.Dropout(0.4))
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

	model:add(nn.SpatialConvolutionMM(200, 250, 2, 2))
	model:add(nn.Dropout(0.5))

	model:add(nn.SpatialConvolutionMM(250, 10, 1, 1))
	model:add(nn.Reshape(10))
	model:add(nn.SoftMax())		

	----------------------------------------------------------------------
	print '==> here is the model:'
	print(model)

	----------------------------------------------------------------------
	return model
end
