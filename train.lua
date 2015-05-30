require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining some tools'

if opt.loss == 'nll' then
  criterion = nn.ClassNLLCriterion()
elseif opt.loss == 'mse' then
  --model:add(nn.Tanh())
  criterion = nn.MSECriterion()
  criterion.sizeAverage = false  
  if trainData then
    -- convert training labels:
    local trsize = (#trainData.labels)[1]
    noutputs = 10
    local trlabels = torch.CudaTensor( trsize, noutputs )
    trlabels:fill(-1)
    for i = 1,trsize do
       trlabels[{ i,trainData.labels[i] }] = 1
    end
    trainData.labels = trlabels

    -- convert test labels
    local tesize = (#testData.labels)[1]
    local telabels = torch.CudaTensor( tesize, noutputs )
    telabels:fill(-1)
    for i = 1,tesize do
       telabels[{ i,testData.labels[i] }] = 1
    end
    testData.labels = telabels
  end

end

-- CUDA?
if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end
print '==> here is the loss function:'
print(criterion)
-- classes
classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
            'frog', 'horse', 'ship', 'truck'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
	parameters, gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

if opt.optimization == 'CG' then
	optimState = {
		maxIter = opt.maxIter
	}
	optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
	optimState = {
		learningRate = opt.learningRate,
		maxIter = opt.maxIter,
		nCorrection = 10
	}
	optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
	optimState = {
		learningRate = opt.learningRate,
		weightDecay = opt.weightDecay,
		momentum = opt.momentum,
		learningRateDecay = 1e-7
	}
	optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
	optimState = {
		eta0 = opt.learningRate,
		t0 = trsize * opt.t0
	}
	optimMethod = optim.asgd

else
	error('unknown optimization method')
end

----------------------------------------------------------------------
print '==> defining training procedure'

function train()
	epoch = epoch or 1
	local time = sys.clock()

	-- set model to training mode (for modules that differ in training and testing, like Dropout)
	model:training()

	-- shuffle at each epoch
	shuffle = torch.randperm(trsize)

	-- do one epoch
	print('==> doing epoch on training data:')
	print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for t = 1,trainData:size(),opt.batchSize do
		-- disp progress
		xlua.progress(t, trainData:size())

		-- create mini batch
		local inputs = {}
		local targets = {}
		for i = t, math.min(t+opt.batchSize-1, trainData:size()) do
			-- load new sample
			local input = trainData.data[shuffle[i]]
			local target = trainData.labels[shuffle[i]]      
			if opt.type == 'double' then 
				input = input:double()
			elseif opt.type == 'cuda' then 
				input = input:cuda() 
			end
			table.insert(inputs, input)
			table.insert(targets, target)
		end

		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
			-- get new parameters
			if x ~= parameters then
			  parameters:copy(x)
			end

			-- reset gradients
			gradParameters:zero()

			-- f is the average of all criterions
			local f = 0

			-- evaluate function for complete mini batch
			for i = 1, #inputs do
				-- estimate f				
				local output = model:forward(inputs[i])                          
				local err = criterion:forward(output, targets[i])
				f = f + err

				-- estimate df/dW
				local df_do = criterion:backward(output, targets[i])
				model:backward(inputs[i], df_do)

				-- update confusion
				confusion:add(output, targets[i])
			end

			-- normalize gradients and f(X)
			gradParameters:div(#inputs)
			f = f/#inputs

			-- return f and df/dX
			return f,gradParameters
		end

		-- optimize on current mini-batch
		if optimMethod == optim.asgd then
			_,_,average = optimMethod(feval, parameters, optimState)
		else         
			optimMethod(feval, parameters, optimState)
		end
	end

	-- time taken
	time = sys.clock() - time
	time = time / trainData:size()
	print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

	-- print confusion matrix
	print(confusion)

	-- update logger/plot
	trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
	if opt.plot then
		trainLogger:style{['% mean class accuracy (train set)'] = '-'}
		trainLogger:plot()
	end

	-- save/log current net
	local filename = paths.concat(opt.save, 'model.net')
	os.execute('mkdir -p ' .. sys.dirname(filename))
	print('==> saving model to '..filename)
	torch.save(filename, model)

	-- next epoch
	confusion:zero()
	epoch = epoch + 1
end
