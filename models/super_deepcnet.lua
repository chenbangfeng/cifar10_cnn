require 'torch'   -- torch
require 'image'   -- for image transforms
require 'cunn'      -- provides all sorts of trainable modules/layers

function init_model()
	print '==> construct model'

	--Dr. Ben Graham deep model on 126x126 data	
	model = nn.Sequential()

	model:add(nn.SpatialConvolutionMM(3, 320, 2, 2))
	model:add(nn.ReLU())
  model:add(nn.SpatialConvolutionMM(320, 320, 2, 2))
	model:add(nn.ReLU())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  model:add(nn.SpatialConvolutionMM(320, 640, 2, 2))
	model:add(nn.ReLU())
  model:add(nn.Dropout(0.1))
  model:add(nn.SpatialConvolutionMM(640, 640, 2, 2))
	model:add(nn.ReLU())
  model:add(nn.Dropout(0.1))
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  
  model:add(nn.SpatialConvolutionMM(640, 960, 2, 2))
	model:add(nn.ReLU())
  model:add(nn.Dropout(0.2))
  model:add(nn.SpatialConvolutionMM(960, 960, 2, 2))
	model:add(nn.ReLU())
  model:add(nn.Dropout(0.2))
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  
	model:add(nn.SpatialConvolutionMM(960, 1280, 2, 2))
	model:add(nn.ReLU())
  model:add(nn.Dropout(0.3))
  model:add(nn.SpatialConvolutionMM(1280, 1280, 2, 2))
	model:add(nn.ReLU())
  model:add(nn.Dropout(0.3))
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  
  model:add(nn.SpatialConvolutionMM(1280, 1600, 2, 2))
	model:add(nn.ReLU())
  model:add(nn.Dropout(0.4))
  model:add(nn.SpatialConvolutionMM(1600, 1600, 2, 2))
	model:add(nn.ReLU())
  model:add(nn.Dropout(0.4))
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  
  model:add(nn.SpatialConvolutionMM(1600, 1920, 2, 2))
	model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolutionMM(1920, 1920, 1, 1))
	model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))

	model:add(nn.SpatialConvolutionMM(1920, 10, 1, 1))
	model:add(nn.Reshape(10))
	model:add(nn.SoftMax())		

	----------------------------------------------------------------------
	print '==> here is the model:'
	print(model)

	----------------------------------------------------------------------
	return model
end
