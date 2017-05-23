require 'nn'
require 'image'
require 'InstanceNormalization'
require 'src/utils'
require 'riseml'

local cmd = torch.CmdLine()

cmd:option('-image_size', 0, 'Resize input image to. Do not resize if 0.')
cmd:option('-model', '', 'Path to trained model.')
cmd:option('-cpu', false, 'use this flag to run on CPU')

local params = cmd:parse(arg)

-- Load model and set type
local model = torch.load(params.model)

if params.cpu then 
  tp = 'torch.FloatTensor'
else
  require 'cutorch'
  require 'cunn'
  require 'cudnn'

  tp = 'torch.CudaTensor'
  model = cudnn.convert(model, cudnn)
end

model:type(tp)
model:evaluate()

local function run_image(img_data)
  -- Load image and scale
  local byte_tensor = torch.ByteTensor(torch.ByteStorage():string(img_data))
  local img = image.decompressJPG(byte_tensor, 3):float()

  if params.image_size > 0 then
    img = image.scale(img, params.image_size, params.image_size)
  end

  -- Stylize
  local input = img:add_dummy()
  local stylized = model:forward(input:type(tp)):double()
  stylized = deprocess(stylized[1])

  -- Return
  return image.compressJPG(torch.clamp(stylized,0,1)):storage():string()
end

riseml.serve(run_image)
