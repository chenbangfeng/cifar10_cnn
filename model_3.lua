require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Cifar10 Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'convnet_svhn', 'type of model to construct: linear | mlp | convnet_svhn')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- 10-class problem
noutputs = 10

-- input dimensions
nfeats = 3
width = 32
height = 32
ninputs = nfeats*width*height

-- number of hidden units (for MLP only):
nhiddens = ninputs / 2

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,64,128}
filtsize = 3 --could be 5
poolsize = 2
normkernel = image.gaussian1D(7) -- not necessery

----------------------------------------------------------------------
print '==> construct model'

if opt.model == 'linear' then

   -- Simple linear model
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,noutputs))

elseif opt.model == 'mlp' then

   -- Simple 2-layer neural network, with tanh hidden units
   model = nn.Sequential()
   model:add(nn.Reshape(ninputs))
   model:add(nn.Linear(ninputs,nhiddens))
   model:add(nn.Tanh())
   model:add(nn.Linear(nhiddens,noutputs))

elseif opt.model == 'convnet_svhn' then
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

      -- stage 3 : standard 2-layer neural network
      -- with filter size = 3 and poolsize = 2
      -- the size (w, h) of the images after 2 conv+pool layers above is
      -- 32->30->15->13->6
      image_size = 6	
      model:add(nn.View(nstates[2]*image_size*image_size))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(nstates[2]*image_size*image_size, nstates[3]))
      model:add(nn.ReLU())      
      model:add(nn.Linear(nstates[3], noutputs))
      model:add(nn.LogSoftMax())

else

   error('unknown -model')

end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
-- Visualization is quite easy, using itorch.image().
function visualize()
	if opt.visualize then
	   if opt.model == 'convnet_svhn' then
	      if itorch then
		 print '==> visualizing ConvNet filters'
		 print('Layer 1 filters:')
		 itorch.image(model:get(1).weight)
		 print('Layer 2 filters:')
		 itorch.image(model:get(4).weight)
	      else
		 print '==> To visualize filters, start the script in itorch notebook'
	      end
	   end
	end
end
