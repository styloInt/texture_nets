local PairwiseEuclidean, parent = torch.class("nn.PairwiseEuclidean", "nn.Module")

function PairwiseEuclidean:__init()
    parent.__init(self)
    self.gradInput = {torch.Tensor(), torch.Tensor()}
    self.tmp = torch.Tensor()
end


function PairwiseEuclidean:updateOutput(input)
    assert(input[1]:size(2) == input[2]:size(2), 'dim mismatch')

    local B1 = input[1]:size(1)
    local B2 = input[2]:size(1)

    local input1_norm = torch.cmul(self.tmp, input[1], input[1]):sum(2):repeatTensor(1, B2)
    local input2_norm = torch.cmul(self.tmp, input[2], input[2]):sum(2):repeatTensor(1, B1)

    torch.addmm(self.output, 1, input1_norm:add(input2_norm:transpose(1, 2)), - 2, input[1], input[2]:t())
    return self.output
end

function PairwiseEuclidean:updateGradInput(input, gradOutput)
  assert(input[1]:size(2) == input[2]:size(2),   'dim mismatch')
  assert(input[1]:size(1) == gradOutput:size(1), 'dim mismatch')
  assert(input[2]:size(1) == gradOutput:size(2), 'dim mismatch')

  self.gradInput[1]:resizeAs(input[1])
  self.gradInput[2]:resizeAs(input[2])

  local gradOutput_t = gradOutput:t():contiguous()

  torch.addmm(self.gradInput[1], 2, gradOutput  :sum(2):repeatTensor(1, input[1]:size(2)):cmul(input[1]),  -2, gradOutput  , input[2])
  torch.addmm(self.gradInput[2], 2, gradOutput_t:sum(2):repeatTensor(1, input[2]:size(2)):cmul(input[2]),  -2, gradOutput_t, input[1])

  return self.gradInput
end
