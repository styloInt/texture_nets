require 'src/content_loss'
require 'src/texture_loss'

require 'loadcaffe'

local ArtisticCriterion, parent = torch.class('nn.ArtisticCriterion', 'nn.Criterion')

function ArtisticCriterion:__init(params, cnn, texture_image)
   parent.__init(self)

   self.content_modules = {}
   self.texture_modules = {}
   self.descriptor_net = create_descriptor_net(params, cnn, texture_image)
   self:updateModules()
   self.gradInput = nil
end

function ArtisticCriterion:updateOutput(input, target)
  -- Compute target content features

  if #self.content_modules > 0 then
    for k, module in pairs(self.texture_modules) do
      module.active = false
    end
    for k, module in pairs(self.content_modules) do
      module.active = false
    end
    self.descriptor_net:forward(target)
  end
  
  -- Now forward with images from generator
  for k, module in pairs(self.texture_modules) do
    module.active = true
  end
  for k, module in pairs(self.content_modules) do
    module.active = true
    module.target:resizeAs(module.output)
    module.target:copy(module.output)
  end
  self.descriptor_net:forward(input)
  
  local loss = 0
  for _, mod in ipairs(self.content_modules) do
    loss = loss + mod.loss
  end
  for _, mod in ipairs(self.texture_modules) do
    loss = loss + mod.loss
  end

  return loss
end

function ArtisticCriterion:updateGradInput(input, target)
  self.gradInput = self.descriptor_net:updateGradInput(input, target)
  return self.gradInput
end

function ArtisticCriterion:updateModules()
  self.content_modules = {}
  self.texture_modules = {}
  for key,module in pairs(self.descriptor_net.modules) do
    if module.tag == 'content' then
      module.target:cuda();
      table.insert(self.content_modules, module)
    elseif module.tag == 'texture' then
      module.target:cuda();
      table.insert(self.texture_modules, module)
    end
  end
end

function ArtisticCriterion:replace(f)
  self.descriptor_net = self.descriptor_net:replace(f)
  self:updateModules()
  return self
end

function nop()
  -- nop.  not needed by our net
end


function create_descriptor_net(params, cnn, texture_image)

  local content_layers = params.content_layers:split(",") 
  local texture_layers  = params.texture_layers:split(",")

  -- Set up the network, inserting texture and content loss modules
  local content_modules, texture_modules = {}, {}
  local next_content_idx, next_texture_idx = 1, 1
  local net = nn.Sequential()

  for i = 1, #cnn do
     if next_content_idx <= #content_layers or next_texture_idx <= #texture_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')

      if params.vgg_no_pad and (layer_type == 'nn.SpatialConvolution' or layer_type == 'nn.SpatialConvolutionMM' or layer_type == 'cudnn.SpatialConvolution') then
          print (name, ': padding set to 0')

          layer.padW = 0 
          layer.padH = 0 
      end
      net:add(layer)
   
      ---------------------------------
      -- Content
      ---------------------------------
      if name == content_layers[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)

        local norm = false
        local loss_module = nn.ContentLoss(params.content_weight, norm);
        net:add(loss_module)
        loss_module.tag = 'content'
        next_content_idx = next_content_idx + 1
      end
      ---------------------------------
      -- Texture
      ---------------------------------
      if name == texture_layers[next_texture_idx] then
        print("Setting up texture layer  ", i, ":", layer.name)
        local gram = GramMatrix():float();

        local target_features = net:forward(texture_image:add_dummy()):clone()
        local target = gram:forward(nn.View(-1):float():setNumInputDims(2):forward(target_features[1])):clone()

        target:div(target_features[1]:nElement())

        local norm = params.normalize_gradients
        local loss_module = nn.TextureLoss(params.texture_weight, target, norm):float();
        
        net:add(loss_module)
        loss_module.tag = 'texture';
        next_texture_idx = next_texture_idx + 1
      end
    end
  end

  net:add(nn.DummyGradOutput())
  
  -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' or torch.type(module) == 'nn.SpatialConvolution' or torch.type(module) == 'cudnn.SpatialConvolution' then
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  net = cudnn.convert(net, cudnn):cuda()
  collectgarbage()
  return net, content_modules, texture_modules
end
