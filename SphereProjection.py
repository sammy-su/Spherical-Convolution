
import numpy as np
from scipy.sparse import csr_matrix

class SphereCoordinates(object):
    def __init__(self, kernel_size=3, sphereW=1280, sphereH=640, view_angle=65.5, imgW=640):
        '''
        Assume even -- sphereH / sphereW / imgW
        Assume odd -- kernel_size
        '''

        self.sphereW = sphereW
        self.sphereH = sphereH
        self.kernel_size = kernel_size
        self.shape = (kernel_size, kernel_size)

        TX, TY = self._meshgrid()
        kernel_angle = kernel_size * view_angle / imgW
        R, ANGy = self._compute_radius(kernel_angle, TY)

        self._R = R
        self._ANGy = ANGy
        self._Z = TX

    def _meshgrid(self):
        TX, TY = np.meshgrid(range(self.kernel_size), range(self.kernel_size))

        center = self.kernel_size / 2
        if self.kernel_size % 2 == 1:
            TX = TX.astype(np.float64) - center
            TY = TY.astype(np.float64) - center
        else:
            TX = TX.astype(np.float64) + 0.5 - center
            TY = TY.astype(np.float64) + 0.5 - center
        return TX, TY

    def _compute_radius(self, angle, TY):
        _angle = np.pi * angle / 180.
        r = self.kernel_size/2 / np.tan(_angle/2)
        R = np.sqrt(np.power(TY, 2) + r**2)
        ANGy = np.arctan(-TY/r)
        return R, ANGy

    def generate_grid(self, tilt):
        if not self.sphereH > tilt >= 0:
            raise ValueError("Invalid polar displace")
        rotate_y = (self.sphereH/2 - 0.5 - tilt) * np.pi / self.sphereH
        Px, Py = self._sample_points(rotate_y)
        return Px, Py

    def _sample_points(self, rotate_y):
        angle_x, angle_y = self._direct_camera(rotate_y)
        # align center pixel with pixel on the image
        Px = (angle_x + np.pi) / (2*np.pi) * self.sphereW
        Py = (np.pi/2 - angle_y) / np.pi * self.sphereH - 0.5

        # Assume dead zone on the pole
        INDy = Py < 0
        Py[INDy] = 0
        INDy = Py > self.sphereH - 1
        Py[INDy] = self.sphereH - 1

        # check boundary, ensure interpolation
        INDx = Px < 0
        Px[INDx] += self.sphereW
        INDx = Px >= self.sphereW
        Px[INDx] -= self.sphereW
        return Px, Py

    def _direct_camera(self, rotate_y):
        angle_y = self._ANGy + rotate_y
        INDn = np.abs(angle_y) > np.pi/2 # Padding great circle

        X = np.sin(angle_y) * self._R
        Y = - np.cos(angle_y) * self._R
        Z = self._Z

        angle_x = np.arctan(Z / -Y)
        # Padding great circle leads to unsymmetric receptive field
        # so pad with neighbor pixel
        angle_x[INDn] += np.pi
        RZY = np.linalg.norm(np.stack((Y, Z), axis=0), axis=0)
        angle_y = np.arctan(X / RZY)

        INDx = angle_x <= -np.pi
        angle_x[INDx] += 2*np.pi
        INDx = angle_x > np.pi
        angle_x[INDx] -= 2*np.pi
        return angle_x, angle_y


class SphereProjection(SphereCoordinates):
    def __init__(self, kernel_size=3, sphereW=1280, sphereH=640, view_angle=65.5, imgW=640):
        super(SphereProjection, self).__init__(kernel_size, sphereW, sphereH, view_angle, imgW)

    def buildP(self, tilt):
        Px, Py = self.generate_grid(tilt)
        row = []
        col = []
        data = []
        for oy in xrange(Px.shape[0]):
            for ox in xrange(Px.shape[1]):
                ix = Px[oy, ox]
                iy = Py[oy, ox]
                c00, c01, c10, c11 = self._bilinear_coef(ix, iy)
                i00, i01, i10, i11 = self._bilinear_idx(ix, iy)
                oi = oy * Px.shape[1] + ox

                row.append(oi)
                col.append(i00)
                data.append(c00)

                row.append(oi)
                col.append(i01)
                data.append(c01)

                row.append(oi)
                col.append(i10)
                data.append(c10)

                row.append(oi)
                col.append(i11)
                data.append(c11)
        P = csr_matrix((data, (row, col)), shape=(Px.size, self.sphereH*self.sphereW))
        return P

    def _bilinear_coef(self, ix, iy):
        ix0, ix1, iy0, iy1 = self._compute_coord(ix, iy)
        dx0 = ix - ix0
        dx1 = ix1 - ix
        dy0 = iy - iy0
        dy1 = iy1 - iy
        c00 = dx1 * dy1
        c01 = dx1 * dy0
        c10 = dx0 * dy1
        c11 = dx0 * dy0
        return c00, c01, c10, c11

    def _bilinear_idx(self, ix, iy):
        ix0, ix1, iy0, iy1 = self._compute_coord(ix, iy)
        if ix > self.sphereW - 1:
            if ix > self.sphereW:
                raise ValueError("Invalid X index")
            ix1 = 0
        if iy1 >= self.sphereH:
            iy1 = self.sphereH - 1
        if iy0 <= 0:
            iy0 = 0

        i00 = iy0 * self.sphereW + ix0
        i10 = iy0 * self.sphereW + ix1
        i01 = iy1 * self.sphereW + ix0
        i11 = iy1 * self.sphereW + ix1
        return i00, i01, i10, i11

    def _compute_coord(self, ix, iy):
        if ix.is_integer():
            ix0 = int(ix)
            ix1 = ix0 + 1
        else:
            ix0 = int(np.floor(ix))
            ix1 = int(np.ceil(ix))
        if iy.is_integer():
            iy0 = int(iy)
            iy1 = iy0 + 1
        else:
            iy0 = int(np.floor(iy))
            iy1 = int(np.ceil(iy))
        return ix0, ix1, iy0, iy1

    def project(self, P, img):
        output = np.stack([P.dot(img[:,:,c].ravel()).reshape(self.shape) for c in xrange(3)], axis=2)
        return output


def crop_image(src, x, y, crop_size):
    crop_w, crop_h = crop_size
    h, w, c = src.shape
    dst = np.zeros((crop_h, crop_w, c), dtype=src.dtype)

    for i in xrange(crop_h):
        dy = y + i - crop_h / 2
        if dy < 0:
            dy = -1 - dy
            shift_x = w / 2
        elif dy >= h:
            dy = h - 1 - (dy - h)
            shift_x = w / 2
        else:
            shift_x = 0
        for j in xrange(crop_w):
            dx = x + j - crop_w / 2 + shift_x
            if dx >= w:
                dx -= w
            elif dx < 0:
                dx += w
            dst[i,j,:] = src[dy,dx,:]
    return dst


