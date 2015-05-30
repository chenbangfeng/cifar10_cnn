require 'torch'   -- torch
require 'image'   -- for image transforms
require 'cunn'      -- provides all sorts of trainable modules/layers


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

	-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
	model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

	-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
	model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

	-- stage 3 : filter bank -> squashing -> L2 pooling -> normalization
	model:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], filtsize, filtsize))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
	-- stage 4 : standard 2-layer neural network
	-- with filter size = 3 and poolsize = 2
	-- the size (w, h) of the images after 2 conv+pool layers above is
	-- 32->30->15->13->6->4->2
	image_size = 2	
	model:add(nn.View(nstates[3]*image_size*image_size))
	model:add(nn.Dropout(0.5))
	model:add(nn.Linear(nstates[3]*image_size*image_size, nstates[4]))
	model:add(nn.ReLU())
	model:add(nn.Linear(nstates[4], noutputs))
	model:add(nn.LogSoftMax())

	----------------------------------------------------------------------
	print '==> here is the model:'
	print(model)

	----------------------------------------------------------------------
	return model
end
