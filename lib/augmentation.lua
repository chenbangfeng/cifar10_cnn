require 'image'

function enlarge_data(ori_width, ori_height, width, height, background)
	--Put the original image into the center of the canvas size width*height
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

function data_augmentation_hflip(data_ori, w, h, aug_size)  
  local ori_size = data_ori:size()  
  local data_aug = {
     data = torch.Tensor(ori_size + aug_size, 3, w, h),
     labels = torch.Tensor(ori_size + aug_size),
     size = function() return ori_size + aug_size end
  }
  data_aug.data[{ {1, ori_size} }] = data_ori.data
  data_aug.labels[{ {1, ori_size} }] = data_ori.labels
  for i = 1, aug_size do
    if i > ori_size then break end
    data_aug.data[{ {ori_size + i} }] = image.hflip(data_ori.data[{i}])
    data_aug.labels[{ {ori_size + i} }] = data_ori.labels[{ {i} }]
  end  
  return data_aug
end

function image_translate(img, x, y, bgr_color)  
  local channels = img:size()[1]
  local w = img:size()[2]
  local h = img:size()[3]
  local tr_img = torch.Tensor(channels, w, h)
  tr_img:fill(bgr_color)  
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

function data_augmentation_translate(data_ori, w, h, p, aug_size)
  --translate x pixels horizontally, y pixels vertically
  --p is the range of translation    
  local ori_size = data_ori:size()  
  local data_aug = {
     data = torch.Tensor(ori_size + aug_size, 3, w, h),
     labels = torch.Tensor(ori_size + aug_size),
     size = function() return ori_size + aug_size end
  }
  data_aug.data[{ {1, ori_size} }] = data_ori.data
  data_aug.labels[{ {1, ori_size} }] = data_ori.labels
  
  for i = 1, aug_size do
    if i > ori_size then break end
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
    data_aug.data[{ {ori_size + i} }] = image_translate(data_ori.data[{i}], x, y, 128)
    data_aug.labels[{ {ori_size + i} }] = data_ori.labels[{ {i} }]
  end  
  return data_aug
end

function data_augmentation_rotate(data_ori, w, h, pl, ph, aug_size)
  local ori_size = data_ori:size()  
  local data_aug = {
     data = torch.Tensor(ori_size + aug_size, 3, w, h),
     labels = torch.Tensor(ori_size + aug_size),
     size = function() return ori_size + aug_size end
  }
  data_aug.data[{ {1, ori_size} }] = data_ori.data
  data_aug.labels[{ {1, ori_size} }] = data_ori.labels
  local alpha
  for i = 1, aug_size do
    alpha = torch.uniform(pl, ph)
    print(data_ori.data[i])
    print(alpha)
    data_aug.data[{ {ori_size + i} }] = image.rotate(data_ori.data[i], alpha)
    data_aug.labels[{ {ori_size + i} }] = data_ori.labels[{ {i} }]    
  end
  
  return data_aug
end

function affine_transform(data_ori, aug_size, w, h, trl, trh, anl, anh) 
  local ori_size = data_ori:size()  
  local data_aug = {
     data = torch.Tensor(ori_size + aug_size, 3, w, h),
     labels = torch.Tensor(ori_size + aug_size),
     size = function() return ori_size + aug_size end
  }
  data_aug.data[{ {1, ori_size} }] = data_ori.data
  data_aug.labels[{ {1, ori_size} }] = data_ori.labels  
  for i = 1, aug_size do
    ori_i = i % ori_size
    if ori_i == 0 then ori_i = ori_size end
    local alpha = torch.uniform(anl, anh)
    local x = torch.random(trl, trh)
    local y = torch.random(trl, trh)
    local flip = torch.random(2)    
    data_aug.labels[{ {ori_size + i} }] = data_ori.labels[{ {ori_i} }]
    local aff_img = data_ori.data[ori_i]
    if flip == 1 then aff_img = image.hflip(aff_img) end
    aff_img = image.rotate(aff_img, alpha)
    aff_img = image.translate(aff_img, x, y)
    data_aug.data[{ {ori_size + i} }] = aff_img
  end
  return data_aug
end

function data_augmentation(data, w, h, aug_size, tr, alpha)
  local orisize = data:size()
  w = w or data.data:size(3)
  h = h or data.data:size(4)
  aug_size = aug_size or data.data:size(1)*3
  tr = tr or 5
  alpha = alpha or 0.2
  print(data:size())  
  data = affine_transform(data, aug_size, w, h, -tr, tr, -alpha, alpha)
  --[[
  data = data_augmentation_hflip(data, w, h, orisize)
  print(data:size())       
  data = data_augmentation_translate(data, w, h, 20, orisize)
  print(data:size())    
  data = data_augmentation_rotate(data, w, h, -0.2, 0.2, orisize)
  print(data:size())    
  ]]  
  return data
end