import pydiffvg
import torch
import skimage.io

# 使用 GPU
device = torch.device('cuda')
pydiffvg.set_use_gpu(torch.cuda.is_available())

# 创建一个 256x256 的画布
canvas_width = 256
canvas_height = 256
shapes = []
shape_groups = []

# 定义一个圆
circle = pydiffvg.Circle(radius=torch.tensor(100.0),
                         center=torch.tensor([128.0, 128.0]))
shapes.append(circle)

# 定义红色填充
circle_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]),
                                   fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0]))
shape_groups.append(circle_group)

# 渲染
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

render = pydiffvg.RenderFunction.apply
img = render(256, 256, 2, 2, 0, None, *scene_args)

# ==== 修改开始 ====
import numpy as np

# 1. 拿到数据
img_numpy = img.cpu().numpy()

# 2. 转换类型：从 0.0-1.0 (float) 转为 0-255 (uint8)
# 为了防止数值溢出，先限制在 0-1 之间，然后乘以 255，最后转整数
img_uint8 = (img_numpy.clip(0, 1) * 255).astype(np.uint8)

# 3. 保存
skimage.io.imsave("test_circle.png", img_uint8)
# ==== 修改结束 ====

print("Successfully rendered test_circle.png!")