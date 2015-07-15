require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers


function init_model()	
	print '==> construct model'
		
	model = nn.Sequential()
	
	model:add(nn.SpatialConvolutionMM(3, 64, 3, 3, 1, 1, 1))
	model:add(nn.PReLU())
  model:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1))
	model:add(nn.PReLU())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.Dropout(0.25))
  
  model:add(nn.SpatialConvolutionMM(64, 128, 3, 3, 1, 1, 1))
	model:add(nn.PReLU())
  model:add(nn.SpatialConvolutionMM(128, 128, 3, 3, 1, 1, 1))
	model:add(nn.PReLU())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.Dropout(0.25))
  
  model:add(nn.SpatialConvolutionMM(128, 256, 3, 3, 1, 1, 1))
	model:add(nn.PReLU())
  model:add(nn.SpatialConvolutionMM(256, 256, 3, 3, 1, 1, 1))
	model:add(nn.PReLU())
  model:add(nn.SpatialConvolutionMM(256, 256, 3, 3, 1, 1, 1))
	model:add(nn.PReLU())
  model:add(nn.SpatialConvolutionMM(256, 256, 3, 3, 1, 1, 1))
	model:add(nn.PReLU())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.Dropout(0.25))
  
  --Fully connected
  model:add(nn.SpatialConvolutionMM(256, 1024, 4, 4, 1, 1, 0))
	model:add(nn.PReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolutionMM(1024, 1024, 1, 1, 1, 1, 0))
	model:add(nn.PReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolutionMM(1024, 10, 1, 1))
  model:add(nn.Reshape(10))
  model:add(nn.LogSoftMax())
	
	
	return model
end

function pretrain_weight(pretrain_model)  
  model.modules[1].weights = pretrain_model:get(1).weight
  model.modules[7].weights = pretrain_model:get(5).weight
  model.modules[13].weights = pretrain_model:get(9).weight
  model.modules[15].weights = pretrain_model:get(11).weight  
  
  model.modules[23].weights = pretrain_model:get(15).weight
  model.modules[26].weights = pretrain_model:get(18).weight
end