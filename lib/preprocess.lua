require 'image'
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

function data_augmentation_hflip(w, h, orisize)  
  local augmented_trainData = {
     data = torch.Tensor(trsize + orisize, 3, w, h),
     labels = torch.Tensor(trsize + orisize),
     size = function() return trsize end
  }
  augmented_trainData.data[{ {1, trsize} }] = trainData.data
  augmented_trainData.labels[{ {1, trsize} }] = trainData.labels
  for i = 1, orisize do
    augmented_trainData.data[{ {trsize+i} }] = image.hflip(trainData.data[{i}])
    augmented_trainData.labels[{ {trsize+i} }] = trainData.labels[{ {i} }]
  end
  trsize = trsize + orisize
  trainData = augmented_trainData
end

function image_translate(img, x, y, bgr_color)  
  local channels = img:size()[1]
  local w = img:size()[2]
  local h = img:size()[3]
  local tr_img = torch.Tensor(channels, w, h)
  tr_img:fill(bgr_color)
  --[[
  for i = 1, channels do
    for j = 1, w do
      for k = 1, h do
        tr_img[{{}, {i}, {j}, {k}}] = bgr_color
      end
    end
  end
  --]]
  if x > 0 then
    tr_x = x+1
    im_x = 1
  else
    tr_x = 1
    im_x = torch.abs(x)+1
  end
  
  if y > 0 then
    tr_y = y+1
    im_y = 1
  else
    tr_y = 1
    im_y = torch.abs(y)+1
  end
  local slide_x = w - torch.abs(x) - 1
  local slide_y = h - torch.abs(y) - 1
  for i = 1, channels do
    --print(tr_x, slide_x, tr_y, slide_y)
    --print(im_x, slide_x, im_y, slide_y)
    tr_img[{{i}, {tr_x, tr_x + slide_x}, {tr_y, tr_y + slide_y}}] 
      = img[{{i}, {im_x, im_x + slide_x}, {im_y, im_y + slide_y}}]    
  end
  return tr_img
end

function data_augmentation_translate(w, h, p, orisize)
  --translate x pixels horizontally, y pixels vertically
  --p is the range of translation    
  local augmented_trainData = {
     data = torch.Tensor(trsize + orisize, 3, w, h),
     labels = torch.Tensor(trsize + orisize),
     size = function() return trsize end
  }
  augmented_trainData.data[{ {1,trsize} }] = trainData.data
  augmented_trainData.labels[{ {1,trsize} }] = trainData.labels
  for i = 1, orisize do
    if i%1000 == 0 then
      print(i)
    end
    x = torch.random(p)
    if x > p/2 then
      x = p/2 - x--generate random negative number
    end
    y = torch.random(p)
    if y > p/2 then
      y = p/2 - y
    end
    augmented_trainData.data[{ {trsize+i} }] = image_translate(trainData.data[{i}], x, y, 128)
    augmented_trainData.labels[{ {trsize+i} }] = trainData.labels[{ {i} }]
  end
  trsize = trsize + orisize
  trainData = augmented_trainData  
end

function data_augmentation(w, h)
  local orisize = trsize
  print(trsize)
  print(orisize)
  data_augmentation_hflip(w, h, orisize)
  print(trsize)
  print(orisize)
  print(trainData)
  
  data_augmentation_translate(w, h, 20, orisize)
  print(trsize)
  print(orisize)
  
end