import matplotlib.pyplot as pyplot
import numpy

import matplotlib.transforms as mtransforms


def get_image():
    delta = 0.25
    x = y = numpy.arange(-3.0, 3.0, delta)
    X, Y = numpy.meshgrid(x, y)
    Z1 = numpy.exp(-X ** 2 - Y ** 2)
    Z2 = numpy.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
    Z = (Z1 - Z2)
    return Z


def do_plot(ax, Z, transform):
    im = ax.imshow(Z, interpolation='none',
                   origin='lower',
                   extent=[-2, 4, -3, 2], clip_on=True)

    trans_data = transform + ax.transData
    im.set_transform(trans_data)

    # display intended extent of the image
    x1, x2, y1, y2 = im.get_extent()
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--",
            transform=trans_data)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)


if __name__ == "__main__":
    # prepare image and figure
    fig, ((ax1)) = pyplot.subplots(1, 1)
    Z = get_image()

    # image rotation
    do_plot(ax1, Z, mtransforms.Affine2D().rotate_deg(30))

    pyplot.show()
