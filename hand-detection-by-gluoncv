from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

net = model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes = ['hand', ])
net.load_parameters('E:\\PyCode\\ssd_512_mobilenet1.0_custom-0190.params')

x, img = data.transforms.presets.ssd.load_test(
    'E:\\PyCode\\hand.jpg', short = 512)

class_IDs, scores, bounding_boxes = net(x)

ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0],
                         class_IDs[0], class_names = net.classes)
plt.show()
