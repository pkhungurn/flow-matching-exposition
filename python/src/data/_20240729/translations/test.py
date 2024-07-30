import matplotlib.pyplot as pyplot
import matplotlib.transforms as transforms
import numpy

from data._20240729.constants import HOSHIHINA_600_FILE_NAME
from shion.base.image_util import extract_numpy_image_from_filelike


def get_image():
    image = extract_numpy_image_from_filelike(HOSHIHINA_600_FILE_NAME)
    image = numpy.flip(image, axis=0)
    alpha = numpy.ones(shape=(image.shape[0], image.shape[1], 1))
    image = numpy.concatenate((image, alpha), axis=2)
    return image


def plot_image(ax, image, transform=None, extent=(-1, 1, -1, 1)):
    im = ax.imshow(
        image,
        interpolation='antialiased',
        origin='lower',
        extent=extent,
        clip_on=True)
    if transform is not None:
        trans_data = transform + ax.transData
        im.set_transform(trans_data)


if __name__ == "__main__":
    # prepare image and figure
    fig, ((ax1)) = pyplot.subplots(1, 1)
    image = get_image()

    # image rotation
    #do_plot(ax1, Z, mtransforms.Affine2D().rotate_deg(30))
    plot_image(ax1, image, transforms.Affine2D().rotate_deg(30))

    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.grid()

    pyplot.show()
