require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

function init_model()	
	print '==> construct model'

	model = nn.Sequential()
	--48
	model:add(nn.SpatialConvolutionMM(3, 64, 2, 2))  
	model:add(nn.PReLU())
  	model:add(nn.SpatialConvolutionMM(64, 64, 2, 2))  
	model:add(nn.PReLU())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  	--23
	model:add(nn.SpatialConvolutionMM(64, 128, 2, 2))
	model:add(nn.PReLU())
  	model:add(nn.SpatialConvolutionMM(128, 128, 2, 2))
	model:add(nn.PReLU())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  	model:add(nn.Dropout(0.25))
  	--10
	model:add(nn.SpatialConvolutionMM(128, 256, 2, 2))
	model:add(nn.PReLU())
  	model:add(nn.SpatialConvolutionMM(256, 256, 2, 2))
	model:add(nn.PReLU())
  	model:add(nn.SpatialConvolutionMM(256, 256, 2, 2))
	model:add(nn.PReLU())
  	model:add(nn.SpatialConvolutionMM(256, 256, 2, 2))
	model:add(nn.PReLU())	
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  	model:add(nn.Dropout(0.25))
  	--3
  	model:add(nn.SpatialConvolutionMM(256, 1024, 3, 3))
	model:add(nn.PReLU())
  	model:add(nn.Dropout(0.5))
  	model:add(nn.SpatialConvolutionMM(1024, 1024, 1, 1))
	model:add(nn.PReLU())
  	model:add(nn.Dropout(0.5))

	model:add(nn.SpatialConvolutionMM(1024, 10, 1, 1))
	model:add(nn.Reshape(10))
	model:add(nn.LogSoftMax())			
	return model
end
