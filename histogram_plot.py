
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import errno
import yaml

# IMAGES

# path = "C:\\t\\*.png"
# images = glob.glob(path)
# for image in images:
#     try:
#         pic = mpimg.imread(image)
#         imgplot = plt.imshow(pic)
#         plt.show()
#
#         # print('Type of the image : ', type(pic))
#         # print('Shape of the image : {}'.format(pic.shape))
#         # print('Image Hight {}'.format(pic.shape[0]))
#         # print('Image Width {}'.format(pic.shape[1]))
#         # print('Dimension of Image {}'.format(pic.ndim))
#
#     except IOError as exc:
#         if exc.errno != errno.EISDIR:
#             raise

# META

cad_idx_test = []
cad_idx_train = []

path = "C:\\car\\*.yaml"

path_train = 'C:/car/complete/train/meta/*.yaml'
path_test = 'C:/car/complete/test/meta/*.yaml'

metas_train = glob.glob(path_train)
metas_test = glob.glob(path_test)

for meta in metas_test:
    try:
        with open(meta) as f:
            dataMap = yaml.safe_load(f)
            x = dataMap["cad_idx"]
            # print(dataMap)
            # print(x)
            cad_idx_test.append(x)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

for meta in metas_train:
    try:
        with open(meta) as f:
            dataMap = yaml.safe_load(f)
        x = dataMap["cad_idx"]
        # print(dataMap)
        # print(x)
        cad_idx_train.append(x)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

print(cad_idx_test)
print(cad_idx_train)

plt.hist([cad_idx_train, cad_idx_test], histtype='bar', rwidth=0.8)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
#plt.gca().legend(loc='upper right')
plt.show()
