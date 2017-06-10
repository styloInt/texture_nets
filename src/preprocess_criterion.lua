local PreprocessCriterion, parent = torch.class('nn.PreprocessCriterion', 'nn.Criterion')

function PreprocessCriterion:__init(criterion, preprocessingStep)
   parent.__init(self)
   self.criterion = criterion
   self.preprocessing = preprocessingStep 
   self.input = torch.FloatTensor()
   self.target = torch.FloatTensor()
end

function PreprocessCriterion:updateOutput(input, target)
   if self.preprocessing then
      self.target = self.preprocessing:forward(target):clone()
      self.input = self.preprocessing:forward(input)
   else
      self.target = target
      self.input = input
   end
   self.output = self.criterion:updateOutput(self.input, self.target)
   return self.output
end

function PreprocessCriterion:updateGradInput(input, target)
   local gradInput = self.criterion:updateGradInput(self.input, self.target)
   if self.preprocessing then
      self.gradInput = self.preprocessing:backward(input, gradInput)
   else
      self.gradInput = gradInput
   end
   return self.gradInput
end

function PreprocessCriterion:type(type)
   self.criterion:type(type)
   if self.preprocessing then
      self.preprocessing:type(type)
   end
   return parent.type(self, type)
end

function PreprocessCriterion:replace(f)
   if self.preprocessing then
      self.preprocessing = self.preprocessing:replace(f)
   end
   self.criterion = self.criterion:replace(f)
   return self
end
