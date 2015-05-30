require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

function init_model()
	print '==> define parameters'

	-- 10-class problem
	noutputs = 10

	-- input dimensions
	nfeats = 3
	width = 32
	height = 32
	ninputs = nfeats*width*height	

	-- hidden units, filter sizes (for ConvNet only):
	nstates = {64, 64, 128, 256}
	filtsize = 3 --could be 5
	poolsize = 2
	normkernel = image.gaussian1D(7) -- not necessery

	----------------------------------------------------------------------
	print '==> construct model'
	
	-- a typical modern convolution network (conv+relu+pool)
	model = nn.Sequential()
  final_mlpconv_layer = nil
  
   
  --Small version of nin with 2 convmlp layers

  model:add(nn.SpatialConvolutionMM(3, 64, 3, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolutionMM(64, 64, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.Dropout(0.25))

  model:add(nn.SpatialConvolutionMM(64, 128, 3, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolutionMM(128, 128, 1, 1))--nin
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.Dropout(0.5))

  model:add(nn.SpatialConvolutionMM(128, 256, 3, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolutionMM(256, 256, 1, 1))
  model:add(nn.ReLU())

  -- Global Average Pooling Layer

  --final_mlpconv_layer = nn.SpatialConvolutionMM(256, 10, 1, 1)
  --model:add(final_mlpconv_layer)
  model:add(nn.SpatialConvolutionMM(256, 10, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialAveragePooling(4, 4, 4, 4))
  model:add(nn.Reshape(10))
  model:add(nn.SoftMax())

  -- all initial values in final layer must be a positive number.
  -- this trick is awfully important ('-')b
  --final_mlpconv_layer.weight:abs()
  --final_mlpconv_layer.bias:abs()

	----------------------------------------------------------------------
	print '==> here is the model:'
	print(model)

	----------------------------------------------------------------------
	return model
end
