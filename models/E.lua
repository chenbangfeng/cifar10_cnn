require 'torch'   -- torch
require 'image'   -- for image transforms
require 'cunn'      -- provides all sorts of trainable modules/layers

function init_model()
	print '==> construct model'

	--0 padding on deep_cnet
	model = nn.Sequential()

  	model:add(nn.SpatialZeroPadding(0, 1, 0, 1))
	model:add(nn.SpatialConvolutionMM(3, 64, 2, 2))
	model:add(nn.PReLU())
  	model:add(nn.SpatialZeroPadding(0, 1, 0, 1))
  	model:add(nn.SpatialConvolutionMM(64, 64, 2, 2))
	model:add(nn.PReLU())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  	model:add(nn.SpatialZeroPadding(0, 1, 0, 1))
  	model:add(nn.SpatialConvolutionMM(64, 128, 2, 2))
	model:add(nn.PReLU())
  	model:add(nn.SpatialZeroPadding(0, 1, 0, 1))
  	model:add(nn.SpatialConvolutionMM(128, 128, 2, 2))
	model:add(nn.PReLU())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	model:add(nn.Dropout(0.25))
  
  	model:add(nn.SpatialZeroPadding(0, 1, 0, 1))
  	model:add(nn.SpatialConvolutionMM(128, 256, 2, 2))
	model:add(nn.PReLU())
  	model:add(nn.SpatialZeroPadding(0, 1, 0, 1))
  	model:add(nn.SpatialConvolutionMM(256, 256, 2, 2))
	model:add(nn.PReLU())  	
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
	model:add(nn.Dropout(0.25))
  
	model:add(nn.SpatialZeroPadding(0, 1, 0, 1))
  	model:add(nn.SpatialConvolutionMM(256, 512, 2, 2))
	model:add(nn.PReLU())  	
  	model:add(nn.SpatialZeroPadding(0, 1, 0, 1))
  	model:add(nn.SpatialConvolutionMM(512, 512, 2, 2))
	model:add(nn.PReLU())  	
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))  
	model:add(nn.Dropout(0.25))
  
  	model:add(nn.SpatialConvolutionMM(512, 1024, 2, 2))
	model:add(nn.PReLU())
  	model:add(nn.Dropout(0.5))
  	model:add(nn.SpatialConvolutionMM(1024, 1024, 1, 1))
	model:add(nn.PReLU())
  	model:add(nn.Dropout(0.5))

	model:add(nn.SpatialConvolutionMM(1024, 10, 1, 1))
	model:add(nn.Reshape(10))
	model:add(nn.LogSoftMax())		

	----------------------------------------------------------------------
	print '==> here is the model:'
	print(model)

	----------------------------------------------------------------------
	return model
end
