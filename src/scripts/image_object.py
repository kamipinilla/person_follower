class ImageObject(object):
    """
    Represents an object in an image as a rectangle.

    Attributes:
        x (float): horizontal position of the upper-left corner.
        y (float): vertical position of the upper-left corner.
        w (float): width of the rectangle.
        h (float): height of the rectangle.

        cx (float): horizontal position of the center of the rectangle.
        cy (float): vertical position of the center of the rectangle.
    """
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.cx = x + (w / 2)
        self.cy = y + (h / 2)

    def as_tuple(self):
        """Returns the object as a tuple."""
        return self.x, self.y, self.w, self.h

    def __str__(self):
        return "(x={}, y={}, w={}, h={})".format(self.x, self.y, self.w, self.h)