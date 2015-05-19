function rgb2yuv_transform()
	print '==> preprocessing data'

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
	   std[i] = trainData.data[{ {},i,{},{} }]:std()
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

	-- Define our local normalization operator (It is an actual nn module, 
	-- which could be inserted into a trainable model):
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

	-- It's always good practice to verify that data is properly
	-- normalized.

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

function enlarge_data(ori_width, ori_height, width, height, background)
	--Put the original image into a canvas of size width*height.
	--The background color is put in a grey-scale, 
	--so it is the same for all channels,
	print('==> Start enlarge input data')
	l_trainData = {
	   data = torch.Tensor(trsize, 3, width, height),--:fill(background),
	   labels = torch.Tensor(trsize),
	   size = function() return trsize end
	}

	l_testData = {
	   data = torch.Tensor(tesize, 3, width, height),--:fill(background),
	   labels = torch.Tensor(tesize),
	   size = function() return tesize end
	}

	--l_trainData.data = l_trainData.data:reshape(trsize, 3, width, height)
	--l_testData.data = l_testData.data:reshape(tesize, 3, width, height)

	patch = torch.Tensor(1, 3, width, height)
	for c = 1, 3 do
		for w = 1, width do
			for h = 1, height do
				patch[{{}, {c}, {w}, {h}}] = background
			end
		end
	end
	--print(patch)
	--print(l_trainData.data[{{1},{},{},{}}])
	for i = 1, trsize do
		l_trainData.data[{{i},{},{},{}}] = patch	      	   
	end
	for i = 1, tesize do
		l_testData.data[{{i},{},{},{}}] = patch	      	   
	end
	
	s_w = (width - ori_width)/2
	s_h = (height - ori_height)/2
	l_trainData.data[{{}, {}, {s_w+1, s_w+ori_width}, {s_h+1, s_h+ori_height}}] = trainData.data 
	l_trainData.labels = trainData.labels
	l_testData.data[{{}, {}, {s_w+1, s_w+ori_width}, {s_h+1, s_h+ori_height}}] = testData.data
	l_testData.labels = testData.labels
	trainData = l_trainData
	testData = l_testData
end
