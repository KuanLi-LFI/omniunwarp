'''
This is a python package using Scaramuzza and Mei's method to rectify omnidirectional images

Reference:
1.  https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab
2.  https://docs.opencv.org/4.x/dd/d12/tutorial_omnidir_calib_main.html
'''


import cv2 as cv
import math
import numpy as np
import yaml


class OCAM_MODEL(object):
    '''
    Ocam model template
    '''

    def __init__(self) -> None:
        pass

    def preprocess_img(self, src):
        '''
        Crop the image to select smaller region of interest.
        Current setting is specific to camera and images used in the example.

        Input:
            src (numpy.ndarray) : Input uncropped image
        Output:
            dst (numpy.ndarray) : Cropped image with only omnidirectional FOV
        '''
        cropped = src[80:, 310:1540]

        h, w, c = cropped.shape

        # Erase center oval
        mask1 = np.full((h, w), 255, dtype=np.uint8)
        center1 = (608, 528)
        cv.ellipse(mask1, center1, (208, 250), 0,
                   0, 360, (0, 0, 0), thickness=-1)

        # Erase wheel and red button
        # mask0 = np.full((h, w), 255, dtype=np.uint8)
        # cv.rectangle(mask0, (565, 770), (635, 800),
        #              (0, 0, 0), thickness=-1)
        # cv.rectangle(mask0, (440, 670), (500, 740),
        #              (0, 0, 0), thickness=-1)
        # cv.rectangle(mask0, (720, 670), (775, 752),
        #              (0, 0, 0), thickness=-1)

        # Preserve large circle
        mask2 = np.zeros((h, w), dtype=np.uint8)
        center2 = (600, 505)
        cv.circle(mask2, center2, 490, (255, 255, 255), thickness=-1)

        # mask = cv.bitwise_and(mask, mask1)
        mask = cv.bitwise_and(mask1, mask2)
        self.origin_mask = mask  # mask on circular image

        return cropped, mask

    def rotate_image(self, src, angle, center):
        '''
        Rotate the image by degrees w.r.t. center coordinate

        Input:
            src (numpy.ndarray) : Input image
            angle (float) : Rotation angle in degrees
            center (list[float, float]) : Rotation center pixel coordinate
        Output:
            result (numpy.ndarray) : Rotated image
        '''
        image_center = tuple(np.array(src.shape[1::-1]) / 2)
        rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
        result = cv.warpAffine(
            src, rot_mat, src.shape[1::-1], flags=cv.INTER_NEAREST)
        return result


class SCARA_OCAM_MODEL(OCAM_MODEL):
    '''
    Scaramuzza Ocam model

    The calibration is done in Matlab.
    Default calibration parameters is stored in scara.yaml
    '''

    def __init__(self, **kwargs) -> bool:
        self.xc = 0         # optical center row
        self.yc = 0         # optical center column
        self.c = 0          # affine coefficient
        self.d = 0          # affine coefficient
        self.e = 0          # affine coefficient
        self.invpol = []    # inverse f function coefficients. Used in world2cam
        self.ss = []        # f function coefficients. Used in cam2world
        self.shape = []     # img shape "height" and "width"

        self.lut_90 = None  # look up table for cuboid rectify
        self.lut_pan = None  # look up table for panoramic rectify

        try:
            self.ss = np.array(kwargs['ss'][1:], dtype=np.float64)
            self.invpol = np.array(kwargs['invpol'][1:], dtype=np.float64)
            self.shape = np.array(kwargs['shape'], dtype=np.float64)
            self.c, self.d, self.e = kwargs['cde']
            self.yc, self.xc = kwargs['xy']
        except KeyError as ke:
            print(f"Missing key {ke}")
        except ValueError as ve:
            print(f"Error Format!")
            print(ve)

    def world2cam(self, point3D):
        '''
        Projects 3D point into the image pixel

        Input:
            point3D (list) : [x, y, z] coordinate of 3d points
        Output:
            u (float) : row pixel coordinate
            v (float) : column pixel coordinate
        '''
        invpol = self.invpol
        length_invpol = len(invpol)

        norm = math.sqrt(point3D[0]*point3D[0] + point3D[1]*point3D[1])
        theta = math.atan(point3D[2]/norm)

        if norm != 0:
            invnorm = 1/norm
            t = theta
            rho = invpol[0]
            t_i = 1

            for i in range(1, length_invpol):
                t_i *= t
                rho += t_i*invpol[i]

            x = point3D[0]*invnorm*rho
            y = point3D[1]*invnorm*rho

            u = x * self.c + y * self.d + self.yc
            v = x * self.e + y + self.xc

        else:

            u = self.yc
            v = self.xc

        return u, v

    def create_LUT_90(self, R, H):
        '''
        Create Look Up Table (LUT) for remapping first quadrant omni image into rectanble image
        The LUT only refer to 90 deg FOV of the top-right part (first quadrant) according to the center xc, yc

        (0, 0)
         _________________
        |        | / / / |
        |        | / / / |
        | (xc,yc)| / / / |
        ---------+--------
        |        |       |
        |        |       |
        |        |       |
        ------------------

        Input:
            R (float/int) : The radius of projection cylinder.
            H (float/int) : The height of output image.
        Output:
            mapx (numpy.ndarray) : mapx to be used in np.remap
            mapy (numpy.ndarray) : mapy to be used in np.remap
        '''
        R = int(R)
        H = int(H)
        mapx, mapy = np.zeros(
            (H, 2 * R), dtype=np.float32), np.zeros((H, 2 * R), dtype=np.float32)

        sqrt2 = np.math.sqrt(2)

        for i in range(H):
            for j in range(2*R):
                # Transform the dst image coordinate to XYZ coordinate
                x = sqrt2 / 2 * j
                y = sqrt2 * R - sqrt2 / 2 * j
                z = H / 2 - i

                # Reproject onto image pixel coordinate
                u, v = self.world2cam([-y, x, z])
                mapy[i][j] = u
                mapx[i][j] = v - self.xc  # Top-right, so translate x by xc

        return mapx, mapy

    def create_panoramic_undistortion_LUT(self, Rmin, Rmax, new_img_size):
        '''
        Create Look Up Table (LUT) for remapping omni image into panoramic image

        Input:
            Rmin (int) : Smallest radius to cut from optical center (xc, yc)
            Rmax (int) : Largest radius to cut from optical center (xc, yc)
            new_img_size (list) : size of new image [height, width]
        Output:
            mapx (numpy.ndarray) : mapx to be used in np.remap
            mapy (numpy.ndarray) : mapy to be used in np.remap
        '''
        height = new_img_size[0]
        width = new_img_size[1]

        mapx = np.zeros((height, width), dtype=np.float32)
        mapy = np.zeros((height, width), dtype=np.float32)

        for i in range(height):
            for j in range(width):
                # Note, if you would like to flip the image, just inverte the sign of theta
                theta = (j) / width*2 * math.pi
                rho = Rmax - (Rmax-Rmin) / height * i
                mapx[i][j] = self.xc + rho * math.sin(theta)
                mapy[i][j] = self.yc + rho * math.cos(theta)

        return mapx, mapy

    def panoramic_rectify(self, src, Rmax, Rmin, new_img_size):
        '''
        Rectify omnidirectional image into panoramic image

        Input:
            src (numpy.ndarray) : Input omnidirectional image
            Rmin (int) : Smallest radius to cut from optical center (xc, yc)
            Rmax (int) : Largest radius to cut from optical center (xc, yc)
            new_img_size (list) : size of new image [height, width]
        Output:
            img_rectified (numpy.ndarray) : Rectified panoramic image
        '''

        if not self.lut_pan:
            # Create panoramic look up table
            map_x, map_y = self.create_panoramic_undistortion_LUT(
                Rmax, Rmin, new_img_size)
            self.lut_pan = map_x, map_y
        map_x, map_y = self.lut_pan

        # Remap into panoramic image
        dst = cv.remap(src, map_x, map_y, cv.INTER_NEAREST)
        # Rotate 180 degree to align ground to the bottom
        img_rectified = cv.rotate(dst, cv.ROTATE_180)

        return img_rectified

    def cuboid_rectify(self, src):
        '''
        Rectify omnidirectional image into panoramic image

        Return the perspective images of front, right, back, left and concatenation of four images

        Input:
            src (numpy.ndarray) : Input omnidirectional image
        Output:
            imgs (list[numpy.ndarray]) : Perspective images
            all_image (numpy.ndarray) : Concatenated image
        '''
        # Rotate to align front to the middle
        rotated = self.rotate_image(src, 225, (self.xc, self.yc))

        imgs = []

        if not self.lut_90:

            # The radius of projection cylinder
            R = min((self.xc, self.yc))
            # The height of projection cylinder
            # assuming 30 degrees fov above and below horizon of lens O
            H = R * math.tan(math.radians(30)) * 2

            mapx_90, mapy_90 = self.create_LUT_90(R, H)
            self.lut_90 = mapx_90, mapy_90

        mapx_90, mapy_90 = self.lut_90

        # For each iteration, project the top-right part (first quadrant) according to the center xc, yc
        # Then rotate the image by 90 deg to get the next perspective
        for i in range(4):
            front = rotated[:489, 608:]
            # cv.imwrite(f"front{i}.jpg", front)
            res_90 = cv.remap(front, mapx_90, mapy_90, cv.INTER_NEAREST)
            imgs.append(res_90)
            rotated = self.rotate_image(rotated, 90, (self.xc, self.yc))

        all_img = np.concatenate(imgs, axis=1)

        return imgs, all_img


class MEI_OCAM_MODEL(OCAM_MODEL):
    '''
    Mei Ocam model from opencv. 
    Default calibration parameters is stored in mei.yaml
    '''

    def __init__(self, **kwargs):
        self.K = np.zeros((3, 3))  # Intrinsic Matrix
        self.D = np.zeros((1, 4))  # Distortion coefficients [k1, k2, p1, p2]
        self.Xi = np.zeros((1, 1))  # Mei's Model coefficient

        self.LUT = None

        try:
            self.K = np.array(kwargs['K'], dtype=np.float64).reshape((3, 3))
            self.D = np.array([kwargs['D']], dtype=np.float64)
            self.Xi = np.array([kwargs['Xi']], dtype=np.float64)
        except KeyError as ke:
            print(f"Missing key {ke}")

        except ValueError as ve:
            print(f"Error Format!")
            print(ve)

    def panoramic_rectify(self, src, new_img_size):
        '''
        Use opencv api to rectify omnidirectional image into panoramic image 

        Input:
            src (numpy.ndarray) : Input omnidirectional image 
            new_img_size (list) : size of new image [width, height]
        Output:
            dst (numpy.ndarray) : Rectified panoramic image
        '''
        # Rotated 180 to let the front fit in the middle
        r = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)

        if not self.LUT:
            # Create panoramic look up table
            map1, map2 = cv.omnidir.initUndistortRectifyMap(
                self.K,
                self.D,
                self.Xi,
                r, self.K, new_img_size, cv.CV_16SC2, cv.omnidir.RECTIFY_CYLINDRICAL)
            self.LUT = map1, map2
        map1, map2 = self.LUT

        # Rotated 180 to let the front fit in the middle
        dst = cv.remap(src, map1, map2, cv.INTER_NEAREST)

        return dst


def panoramic_rectify(original_bgr, **kwargs):
    model = kwargs.get('model', 'scara')
    if model == 'scara':
        try:
            scara = SCARA_OCAM_MODEL(ss=kwargs['ss'], invpol=kwargs['invpol'],
                                     shape=kwargs['shape'], cde=kwargs['cde'], xy=kwargs['xy'])
        except KeyError as ke:
            print(f"Missing key {ke}")
            return

        cropped, mask = scara.preprocess_img(original_bgr)
        Rmin = kwargs.get('r2', 210)
        Rmax = kwargs.get('r1', 490)

        # TODO: Find a better cylinder radius, currently is (Rmax + Rmin)s / 2
        res_scara = scara.panoramic_rectify(
            cropped, Rmax, Rmin, (Rmax-Rmin, int((Rmax + Rmin) * np.pi)))
        res_mask = scara.panoramic_rectify(
            mask, Rmax, Rmin, (Rmax-Rmin, int((Rmax + Rmin) * np.pi)))

        return res_scara, res_mask

    elif model == 'mei':
        try:
            mei = MEI_OCAM_MODEL(K=kwargs['K'], D=kwargs['D'], Xi=kwargs['Xi'])
        except KeyError as ke:
            print(f"Missing key {ke}")
            return

        cropped, mask = mei.preprocess_img(original_bgr)
        res_mei = mei.panoramic_rectify(cropped, (2900, 800))
        res_mei_mask = mei.panoramic_rectify(mask, (2900, 800))

        return res_mei, res_mei_mask

    else:
        print("Unsupported Model")
