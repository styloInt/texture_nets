-----Modification of jcjohnson's code -----------
local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
  parent.__init(self)
  print('Using TV loss with weight ', strength)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  if self.strength > 0 then
    self.gradInput:resizeAs(input):zero()
    
    local B, C, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
    self.x_diff = self.x_diff:resize(B, 3, H - 1, W - 1)
    self.y_diff = self.y_diff:resize(B, 3, H - 1, W - 1)

    self.x_diff:copy(input[{{}, {}, {1, -2}, {1, -2}}])
    self.x_diff:add(-1, input[{{}, {}, {1, -2}, {2, -1}}])
    
    self.y_diff:copy(input[{{}, {}, {1, -2}, {1, -2}}])
    self.y_diff:add(-1, input[{{}, {}, {2, -1}, {1, -2}}])
    
    self.gradInput[{{}, {}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
    self.gradInput[{{}, {}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
    self.gradInput[{{}, {}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)

    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)

    return self.gradInput
  else
    self.gradInput = gradOutput
    return self.gradInput
  end
end

function TVLoss:clearState()
  self.output = self.output.new()
  self.gradInput = self.gradInput.new()

  self.x_diff = self.x_diff.new()
  self.y_diff = self.y_diff.new()
end

------------ Criterion ----------------------

-----Modification of jcjohnson's code -----------
local TVCriterion, parent = torch.class('nn.TVCriterion', 'nn.Criterion')

function TVCriterion:__init(strength)
  parent.__init(self)
  print('Using TV loss with weight ', strength)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVCriterion:updateOutput(input, target)
  local B, C, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  self.x_diff = self.x_diff:resize(B, 3, H - 1, W - 1)
  self.y_diff = self.y_diff:resize(B, 3, H - 1, W - 1)

  self.output = self.strength*(torch.norm(self.x_diff) + torch.norm(self.y_diff))
  return self.output
end

function TVCriterion:updateGradInput(input, target)
  self.gradInput:resizeAs(input):zero()
  
  self.x_diff:copy(input[{{}, {}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {}, {1, -2}, {2, -1}}])
  
  self.y_diff:copy(input[{{}, {}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {}, {2, -1}, {1, -2}}])
  
  self.gradInput[{{}, {}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{}, {}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{}, {}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)

  self.gradInput:mul(self.strength)

  return self.gradInput
end

-- function TVCriterion:clearState()
--   self.output = self.output.new()
--   self.gradInput = self.gradInput.new()

--   self.x_diff = self.x_diff.new()
--   self.y_diff = self.y_diff.new()
-- end

-------------

local MultiCriterion1, parent = torch.class('nn.MultiCriterion1', 'nn.Criterion')

function MultiCriterion1:__init()
   parent.__init(self)
   self.criterions = {}
   self.names = {}

   self.weights = torch.DoubleStorage()
end

function MultiCriterion1:add(criterion, weight, name)
   assert(criterion, 'no criterion provided')
   weight = weight or 1
   name = name or #self.names + 1
   table.insert(self.names, name)

   table.insert(self.criterions, criterion)
   self.weights:resize(#self.criterions, true)
   self.weights[#self.criterions] = weight
   return self
end

function MultiCriterion1:updateOutput(input, target)
   self.output = {}
   for i=1,#self.criterions do
      self.output[i] = self.weights[i]*self.criterions[i]:updateOutput(input, target)
   end
   return self.output
end

function MultiCriterion1:updateGradInput(input, target)
   self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
   nn.utils.recursiveFill(self.gradInput, 0)
   for i=1,#self.criterions do
      nn.utils.recursiveAdd(self.gradInput, self.weights[i], self.criterions[i]:updateGradInput(input, target))
   end
   return self.gradInput
end

function MultiCriterion1:type(type)
   for i,criterion in ipairs(self.criterions) do
      criterion:type(type)
   end
   return parent.type(self, type)
end