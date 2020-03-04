import siftgpu
import imageio as io
import numpy as np
from PIL import Image, ImageDraw

x = io.imread('IMG_2759.JPG')

b = siftgpu.Bitmap(x[:, :, ::-1])

print(b.is_grey())

b.write('out.jpg')

b.clone_as_grey().write('out_g.jpg')

options = siftgpu.make_extreme_options()

descriptors, keypoints = siftgpu.extract_covariant_sift(options, b.clone_as_grey())

for keypoint in keypoints:
    print(keypoint.x)
    print(keypoint.y)
    print(keypoint.compute_scale())
    print(keypoint.compute_orientation())

print(descriptors)
print(descriptors.shape)

image = Image.fromarray(x.copy())
draw = ImageDraw.Draw(image)


for keypoint in keypoints:
    point_x = keypoint.x
    point_y = keypoint.y

    draw.line((point_x - 4, point_y - 4, point_x + 4, point_y + 4), fill=(255, 50, 50), width=2)
    draw.line((point_x - 4, point_y + 4, point_x + 4, point_y - 4), fill=(255, 50, 50), width=2)

io.imwrite('sift.jpg', image)

image = Image.fromarray(x.copy())
draw = ImageDraw.Draw(image)

descriptors_new = siftgpu.extract_covariant_sift_given_keypoints(options, b.clone_as_grey(), keypoints[:100])

for keypoint in keypoints:
    point_x = keypoint.x
    point_y = keypoint.y

    draw.line((point_x - 4, point_y - 4, point_x + 4, point_y + 4), fill=(255, 50, 50), width=2)
    draw.line((point_x - 4, point_y + 4, point_x + 4, point_y - 4), fill=(255, 50, 50), width=2)

assert descriptors_new.shape[0] == 100

io.imwrite('sift_from_keypoints.jpg', image)

print(descriptors[:100])
print(descriptors_new)

assert np.all(np.equal(descriptors[:100], descriptors_new))
