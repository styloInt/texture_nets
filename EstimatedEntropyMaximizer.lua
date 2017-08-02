require 'nn'

local EstimatedEntropyMaximizer, parent = torch.class('nn.EstimatedEntropyMaximizer', 'nn.Criterion')
_ = [[
    mode = 'min_dist|all_dist'
]]
function EstimatedEntropyMaximizer:__init(mode)
   parent.__init(self)
   self.mode = mode or 'min_dist'
   self.cdist = nn.PairwiseEuclidean()
end

function EstimatedEntropyMaximizer:updateOutput(input)
   local ov = input:view(input:size(1), -1)
   local dists = self.cdist:forward({ov,ov})

   local d = dists:clone():float()

   if self.mode == 'min_dist' then
    local m = d:max()
    
    -- Fill diag with something
    d = d  + d.new():eye(input:size(1)):mul(m+1)

    -- find mins 
    local _, min_idx = d:min(2)
    min_idx = min_idx:view(-1)

    -- print(d)
    -- print(min_idx)
    self.grad = d:clone():fill(0)
    self.output = 0
    for i = 1, d:size(1) do
      self.output = self.output + math.log(d[i][min_idx[i]])
      self.grad[i][min_idx[i]] = 1/(d[i][min_idx[i]]+1e-1)
    end
    self.output = self.output*ov:size(2)/ov:size(1)
   else

   end
    -- All
    -- for i = 1, d:size(1) do
    --   for j = 1, d:size(2) do
    --     if i == j then 
    --       d[i][j] = 0
    --       dists[i][j] = 1000000000
    --     else
    --       d[i][j] = 1/d[i][j]
    --     end
    --   end
    -- end
   return self.output
end

function EstimatedEntropyMaximizer:updateGradInput(input, target)
   local ov = input:view(input:size(1), -1)
   local cdist_grads = self.cdist:backward({ov,ov}, -self.grad:cuda())

   self.gradInput =  (cdist_grads[1] + cdist_grads[2]):viewAs(input)
   
   return self.gradInput
end



  -- if opt.entropy_loss then 
  --   local ov = out:view(out:size(1), -1)
  --   local dists = cdist:forward({ov,ov})

  --   dists = dists:float()
  --   local d = dists:clone()

  --   -- All
  --   for i = 1, d:size(1) do
  --     for j = 1, d:size(2) do
  --       if i == j then 
  --         d[i][j] = 0
  --         dists[i][j] = 1000000000
  --       else
  --         d[i][j] = 1/d[i][j]
  --       end
  --     end
  --   end

  --   -- Min only
  --   local _, min_idx = dists:min(2)
  --   min_idx = min_idx:view(-1)

  --   d:fill(0)
  --   for i = 1, d:size(1) do
  --     d[i][min_idx[i]] = 1/dists[i][min_idx[i]]
  --     d[min_idx[i]][i] = 1/dists[min_idx[i]][i]
  --   end
  --   -- print(dists:min())
  --   local grad_entopy =  cdist:backward({ov,ov}, -d:cuda())

  --   pixel_net.gradInput:add(0.1, grad_entopy[1])
  --   pixel_net.gradInput:add(0.1, grad_entopy[2])
  -- end