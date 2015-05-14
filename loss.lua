require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Cifar10 Loss Function')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | margin')
   cmd:text()
   opt = cmd:parse(arg or {})

   -- to enable self-contained execution:
   model = nn.Sequential()
end

-- 10-class problem
noutputs = 10

----------------------------------------------------------------------
print '==> define loss'

if opt.loss == 'margin' then

   -- This loss takes a vector of classes, and the index of
   -- the grountruth class as arguments. It is an SVM-like loss
   -- with a default margin of 1.

   criterion = nn.MultiMarginCriterion()

elseif opt.loss == 'nll' then

   -- This loss requires the outputs of the trainable model to
   -- be properly normalized log-probabilities, which can be
   -- achieved using a softmax function

   model:add(nn.LogSoftMax())

   -- The loss works like the MultiMarginCriterion: it takes
   -- a vector of classes, and the index of the grountruth class
   -- as arguments.

   criterion = nn.ClassNLLCriterion()
   
else

   error('unknown -loss')

end

----------------------------------------------------------------------
print '==> here is the loss function:'
print(criterion)
