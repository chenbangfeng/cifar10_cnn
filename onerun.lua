-- second attempt at cifar10 - torch model
require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require './lib/preprocess.lua'
require './lib/load_model.lua'
require './lib/visualize.lua'
require './models/model_3.lua' --specify model used
--require './models/model_nin.lua'
--require './models/model_cnn_deep.lua'
--require './models/model_deepcnet_s.lua'

----------------------------------------------------------------------
-- OPTION
----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Cifar10 Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- data:
cmd:option('-size', 'full', 'how many samples do we load: small | full')
-- loss:
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse')
-- training:
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', true, 'live plot')
cmd:option('-visualize', true, 'visualize weights')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 1e-4, 'weight decay (SGD only)')
cmd:option('-momentum', 0.95, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-preprocess', 'YUV', 'type: none | YUV | enlarge | YUVL')
cmd:option('-loadtype', 'load', 'type: init | load')
cmd:option('-existing_model', 'n', 'y | n')
cmd:option('-test_mode', 'n', 'y | n')
cmd:option('-augmentation', 'n', 'y | n')
cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
-- LOAD DATA
----------------------------------------------------------------------
tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar10.t7.tgz'

if not paths.dirp('cifar-10-batches-t7') then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

print '==> loading dataset'
if opt.loadtype == 'init' then
  if opt.size == 'full' then
    print '==> using regular, full training data'
    trsize = 50000
    tesize = 10000
    last_batch = 4
  elseif opt.size == 'small' then
     print '==> using reduced training data, for fast experiments'
     trsize = 10000
     tesize = 2000
     last_batch = 0
  end

  trainData = {
     data = torch.Tensor(trsize, 3*32*32),
     labels = torch.Tensor(trsize),
     size = function() return trsize end
  }


  for i = 0,last_batch do
     subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')   
     trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
     trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
  end
  trainData.labels = trainData.labels + 1

  subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
  testData = {
     data = subset.data:t():double(),
     labels = subset.labels[1]:double(),
     size = function() return tesize end
  }
  testData.labels = testData.labels + 1

  -- resize dataset (if using small version)
  trainData.data = trainData.data[{ {1,trsize} }]
  trainData.labels = trainData.labels[{ {1,trsize} }]

  testData.data = testData.data[{ {1,tesize} }]
  testData.labels = testData.labels[{ {1,tesize} }]

  -- reshape data                                                                                     
  trainData.data = trainData.data:reshape(trsize,3,32,32)
  testData.data = testData.data:reshape(tesize,3,32,32)  
    
  ----------------------------------------------------------------------
  -- TRANSFORM DATA
  ----------------------------------------------------------------------

  if (opt.preprocess == 'enlarge') 
    or (opt.preprocess == 'YUVL') then
    enlarge_data(32, 32, 48, 48, 128)  
  end
  
  
  ----------------------------------------------------------------------
  -- DATA AUGMENTATION
  ----------------------------------------------------------------------
  if opt.augmentation == 'y' then
    print('==> data augmentation')
    if opt.preprocess == 'YUV' then
      data_augmentation(32, 32)
    elseif opt.preprocess == 'enlarge' then
      data_augmentation(48, 48)
    elseif opt.preprocess == 'YUVL' then
      data_augmentation(48, 48)
    end
    
    print(trainData)  
    print(testData)  
    print(trainData:size())
    print(trsize)
  end
  
  if (opt.preprocess == 'YUV') or (opt.preprocess == 'YUVL') then      
      rgb2yuv_transform()
  end
  ----------------------------------------------------------------------
  -- SAVE DATA
  ----------------------------------------------------------------------
  print('Save transformed data to file')
  if opt.augmentation == 'y' then
    torch.save('data/trainData_aug.dat', trainData)
    torch.save('data/testData_aug.dat', testData)
  elseif opt.preprocess == 'YUV' then
    torch.save('data/trainData_32_YUV.dat', trainData)
    torch.save('data/testData_32_YUV.dat', testData)
  elseif opt.preprocess == 'enlarge' then
    torch.save('data/trainData_48.dat', trainData)
    torch.save('data/testData_48.dat', testData)
  elseif opt.preprocess == 'YUVL' then
    torch.save('data/trainData_48_YUV.dat', trainData)
    torch.save('data/testData_48_YUV.dat', testData)  
  else
    torch.save('data/trainData_ori.dat', trainData)
    torch.save('data/testData_ori.dat', testData)  
  end
  
elseif opt.loadtype == 'load' then
  print('==> load preprocessed data')
  if opt.size == 'full' then
    if opt.augmentation == 'y' then
      trsize = 100000
      tesize = 10000
    else
      trsize = 50000
      tesize = 10000
    end
    if opt.preprocess == 'YUVL' then
      trainData = torch.load('data/trainData_48_YUV.dat')
      testData = torch.load('data/testData_48_YUV.dat')
    else
      trainData = torch.load('data/trainData_32_YUV.dat')
      testData = torch.load('data/testData_32_YUV.dat')
    end
  elseif opt.size == 'small' then
    trsize = 10000
    tesize = 2000
    if opt.preprocess == 'YUVL' then
      trainData = torch.load('data/trainData_48_YUV_small.dat')
      testData = torch.load('data/testData_48_YUV_small.dat')
    else
      trainData = torch.load('data/trainData_32_YUV_small.dat')
      testData = torch.load('data/testData_32_YUV_small.dat')
    end
  end
end
print('Training Data:')
print(trainData)
print()

print('Test Data:')
print(testData)
print()

----------------------------------------------------------------------
-- VISUALIZE DATA
----------------------------------------------------------------------
print '==> visualizing data'

if opt.visualize then
   visualize_input(trainData, 20001, 20051, true)--the first 100 images
end

----------------------------------------------------------------------
-- LOAD MODEL
----------------------------------------------------------------------
highest_test_accuracy = 0
if opt.existing_model == 'n' then
	print('==> Init new model')
	model = init_model()
	--dofile 'loss.lua'
else 
	print('==> Load best test error model')
	--model = load_model('results/results_model3_augmentation/best_test_model.net')
  model = load_model('results/deep_cnet_small/best_test_model.net')
  	--model = load_model('results/results_deepcnet_YUV_regular/best_test_model.net')	
end

dofile 'train.lua'
dofile 'test.lua'
if opt.existing_model == 'y' then
	test()--define highest_test_accuracy
	--print('==>load lastest training model')
	--model = load_model('results/results_model3_YUV_regular/model.net')
end
----------------------------------------------------------------------
-- TRAINING
----------------------------------------------------------------------

if opt.test_mode == 'n' then
  print '==> training!'
  print(model)
  while true do
    train()
    test()
  end
end

