require 'torch'
require 'image'
require 'sys'

-- zca whitening implementation originally from https://github.com/nagadomi/kaggle-cifar10-torch7
local function pcacov(x, means)
   for i = 1, x:size(1) do
      x[i]:add(-1, means)
   end
   local c = torch.mm(x:t(), x)
   for i = 1, x:size(1) do
      x[i]:add(means)
   end
   c:div(x:size(1)-1)
   local ce,cv = torch.symeig(c,'V')
   return ce,cv
end
local function zca_whiten(data, means, P, invP, epsilon)
    local epsilon = epsilon or 1e-5
    local data_size = data:size()
    local dims = data:size()
    local nsamples = dims[1]
    local n_dimensions = data:nElement() / nsamples
    if data:dim() >= 3 then
       data = data:view(nsamples, n_dimensions)
    end
    if not means or not P or not invP then 
      print('calculate mean')
        -- compute mean vector if not provided 
       means = torch.mean(data, 1)
        -- compute transformation matrix P if not provided
      print('calculate pcacov')  
       local ce, cv = pcacov(data, means)
       collectgarbage()
       ce:add(epsilon):sqrt()
       local invce = ce:clone():pow(-1)
       local invdiag = torch.diag(invce)
       print('calculate invdiag')
       P = torch.mm(cv, invdiag)
       P = torch.mm(P, cv:t())

        -- compute inverse of the transformation
       local diag = torch.diag(ce)
       invP = torch.mm(cv, diag)
       invP = torch.mm(invP, cv:t())
    end
    collectgarbage()
    -- remove the means    
    for i = 1, data:size(1) do
       data[i]:add(-1, means)
    end
    -- transform in ZCA space
    if data:size(1) > 100000 then
      -- matrix mul with 16-spliting
      local step = math.floor(data:size(1) / 16)
      for i = 1, data:size(1), step do
        local n = step
        if i + n > data:size(1) then
           n = data:size(1) - i
        end
        if n > 0 then
           data:narrow(1, i, n):copy(torch.mm(data:narrow(1, i, n), P))
        end
        collectgarbage()
      end
    else
      print('copy')
      data:copy(torch.mm(data, P))
    end
    data = data:view(data_size)
    collectgarbage()
    
    return data, means, P, invP
end
local function zca(x, means, P, invP)
   local ax
   ax, means, P, invP = zca_whiten(x, means, P, invP, 0.01) -- 0.1
   print('copy')
   x:copy(ax)
   return means, P, invP
end

local function global_contrast_normalization(x, mean, std)
   local scale = 1.0
   local u = mean or x:mean(1)
   local v = std or (x:std(1):div(scale))
   for i = 1, x:size(1) do
      x[i]:add(-u)
      x[i]:cdiv(v)
   end
   return u, v
end
function zca_transform(x, params)
   params = params or {}
   params['gcn_mean'], params['gcn_std'] = global_contrast_normalization(x, params['gcn_mean'], params['gcn_std'])
   print('zca')
   params['zca_u'], params['zca_p'], params['zca_invp'] = zca(x, params['zca_u'], params['zca_p'], params['zca_invp'])
   
   return params
end

function normalize()
  -- Name channels for convenience
	channels = {'y','u','v'}

	-- Normalize each channel, and store mean/std
	-- per channel. These values are important, as they are part of
	-- the trainable parameters. At test time, test data will be normalized
	-- using these values.
	print '==> preprocessing data: normalize each feature (channel) globally'
	mean = {}
	std = {}
	for i,channel in ipairs(channels) do
	   -- normalize each channel globally:
	   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
     print(mean[i])
	   std[i] = trainData.data[{ {},i,{},{} }]:std()
     print(std[i])
	   trainData.data[{ {},i,{},{} }]:add(-mean[i])
	   trainData.data[{ {},i,{},{} }]:div(std[i])
	end

	-- Normalize test data, using the training means/stds
	for i,channel in ipairs(channels) do
	   -- normalize each channel globally:
	   testData.data[{ {},i,{},{} }]:add(-mean[i])
	   testData.data[{ {},i,{},{} }]:div(std[i])
	end

	-- Local normalization
	print '==> preprocessing data: normalize all three channels locally'

	-- Define the normalization neighborhood:
	neighborhood = image.gaussian1D(13)
	normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

	-- Normalize all channels locally:
	for c in ipairs(channels) do
	   for i = 1,trainData:size() do
	      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
	   end
	   for i = 1,testData:size() do
	      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
	   end
	end

	----------------------------------------------------------------------
	print '==> verify statistics'

	for i,channel in ipairs(channels) do
	   trainMean = trainData.data[{ {},i }]:mean()
	   trainStd = trainData.data[{ {},i }]:std()

	   testMean = testData.data[{ {},i }]:mean()
	   testStd = testData.data[{ {},i }]:std()

	   print('training data, '..channel..'-channel, mean: ' .. trainMean)
	   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

	   print('test data, '..channel..'-channel, mean: ' .. testMean)
	   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
	end
end

function rgb2yuv_transform()
	print '==> YUV transform'

	trainData.data = trainData.data:float()
	testData.data = testData.data:float()

	-- Convert all images to YUV
	print '==> preprocessing data: colorspace RGB -> YUV'
	for i = 1,trainData:size() do
	   trainData.data[i] = image.rgb2yuv(trainData.data[i])    
	end
	for i = 1,testData:size() do
	   testData.data[i] = image.rgb2yuv(testData.data[i])
     
	end

	normalize()
end