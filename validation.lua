--CIFAR-10 classification framework
require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require './lib/augmentation.lua'
require './lib/load_model.lua'
require './lib/visualize.lua'
require './lib/preprocessing.lua'
--choose 1 model from the following models to run
--note that different models may have different setups
require './models/A.lua'
--require './models/B.lua'
--require './models/C.lua'
--require './models/D.lua'
--require './models/E.lua'
--require './models/D2.lua'
--require './models/E2.lua'

----------------------------------------------------------------------
-- OPTION
----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Cifar10 Loss Function')
cmd:text()
cmd:text('Options:')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
cmd:option('-size', 'full', 'how many samples do we load: small | full')
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-plot', true, 'live plot')
cmd:option('-visualize', true, 'visualize weights')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-max_epoch', 30, 'max number of epochs')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0.9, 'momentum (SGD only)')
cmd:option('-type', 'float', 'type: double | float | cuda')
cmd:option('-enlarge', 'n', 'y | n')
cmd:option('-augmentation', 'n', 'y | n')
cmd:option('-preprocess', 'YUV', 'type: none | YUV | ZCA')
cmd:option('-pretrain', 'n', 'y | n')

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
-- INIT DATA
----------------------------------------------------------------------
tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar10.t7.tgz'

if not paths.dirp('cifar-10-batches-t7') then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

print '==> loading dataset'

if opt.size == 'full' then
  print '==> using regular, full training data'
  trsize = 40000
  tesize = 10000
  last_batch = 3
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

--train set
for i = 0, last_batch do
   subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')   
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end
trainData.labels = trainData.labels + 1

--validation set
subset = torch.load('cifar-10-batches-t7/data_batch_5.t7', 'ascii')   
testData = {
   data = subset.data:t():float(),
   labels = subset.labels[1]:float(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

--final test set
subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
final_testData = {
   data = subset.data:t():float(),
   labels = subset.labels[1]:float(),
   size = function() return tesize end
}
final_testData.labels = testData.labels + 1

-- resize dataset (if using small version)
trainData.data = trainData.data[{ {1, trsize} }]
trainData.labels = trainData.labels[{ {1, trsize} }]

testData.data = testData.data[{ {1, tesize} }]
testData.labels = testData.labels[{ {1, tesize} }]

-- reshape data                                                                                     
trainData.data = trainData.data:reshape(trsize, 3, 32, 32)
testData.data = testData.data:reshape(tesize, 3, 32, 32)  
  
----------------------------------------------------------------------
-- TRANSFORM DATA
----------------------------------------------------------------------

if (opt.enlarge == 'y') then
  enlarge_data(32, 32, 48, 48, 0)  
end
  
----------------------------------------------------------------------
-- DATA AUGMENTATION
----------------------------------------------------------------------
if opt.augmentation == 'y' then
  print('==> data augmentation')
  if (opt.enlarge == 'n') then    
    trainData = data_augmentation(trainData, 32, 32)
    testData = data_augmentation(testData, 32, 32)
  else
    trainData = data_augmentation(trainData, 48, 48)
    testData = data_augmentation(testData, 48, 48)
  end    
  trsize = trainData.size()
  tesize = testData.size()
  print(trainData)  
  print(testData)      
end

local params = nil
if (opt.preprocess == 'YUV') then  
  --YUV + contrast normalize
  rgb2yuv_transform()
elseif (opt.preprocess == 'ZCA') then
  ----ZCA + contrast normalize
  print('==> ZCA + contrast normalize')
  print('==> train_set')
  params = zca_transform(trainData.data)
  print('==> test_set')
  zca_transform(testData.data, params)
  collectgarbage()
end
----------------------------------------------------------------------
-- TRAINING
----------------------------------------------------------------------
highest_test_accuracy = 0
print('==> Init new model')
model = init_model()
--[[
if (opt.pretrain == 'y') then
  pretrain_weight('results/A/best_test_model.net')
end
]]
print '==> training!'

dofile 'train.lua'
dofile 'test.lua'

for epoch = 0, opt.max_epoch do
  train()
  test()
end

testData = final_testData
highest_test_accuracy = 0
test()
print("Final testset accuracy: ", highest_test_accuracy)