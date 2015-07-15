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
	
	----------------------------------------------------------------------
	print '==> here is the model:'
	print(model)

	----------------------------------------------------------------------
	return model
end

function pretrain_weight(pretrain_model)
  local conv1 = pretrain_model:get(1).weight
  local conv2 = pretrain_model:get(5).weight
  local conv31 = pretrain_model:get(9).weight
  local conv32 = pretrain_model:get(11).weight
  
  local full1 = pretrain_model:get(15).weight
  local full2 = pretrain_model:get(18).weight
  local full3 = pretrain_model:get(21).weight
  
  model.modules[1].weights = conv1
  model.modules[7].weights = conv2
  model.modules[13].weights = conv31
  model.modules[15].weights = conv32
  model.modules[19].weights = full1
  model.modules[22].weights = full2
  model.modules[25].weights = full3
end