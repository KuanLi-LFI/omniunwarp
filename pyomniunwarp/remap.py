'''
This is a python package using Scaramuzza and Mei's method to rectify omnidirectional images

Reference:
1.  https://sites.google.com/site/scarabotix/ocamcalib-omnidirectional-camera-calibration-toolbox-for-matlab
2.  https://docs.opencv.org/4.x/dd/d12/tutorial_omnidir_calib_main.html
'''

import cv2 as cv
import math
import numpy as np


class ScaramuzzaOcamModel():
    '''
    Scaramuzza Ocam model instance.
    Including model and unwarp projection functions.

    The calibration is done in Matlab.
    Default calibration parameters is stored in calib_results.txt
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

        self.Rmax = 600     # Outer ring radius default value
        self.Rmin = 200     # Inner ring radius default value

        self.LUT_cylinder = None    # look up table for cylinder rectify
        self.LUT_cuboid = None      # look up table for cuboid rectify

        self.mask_ring = None       # mask of ring image
        # mask of cuboid image [front, left, back, right, full, front-left, front-right]
        self.mask_cuboid_plus = None
        self.mask_cylinder = None   # mask of cylinder image

        # Read parameters
        try:
            self.ss = np.array(kwargs['param']['ss'][1:], dtype=np.float64)
            self.invpol = np.array(
                kwargs['param']['invpol'][1:], dtype=np.float64)
            self.shape = np.array(kwargs['param']['shape'], dtype=np.float64)
            self.c, self.d, self.e = kwargs['param']['cde']
            self.yc, self.xc = kwargs['param']['xy']
            self.Rmax, self.Rmin = kwargs['param']['R']
        except KeyError as ke:
            print(f"Missing key {ke}")
        except ValueError as ve:
            print(f"Error Format!")
            print(ve)

    # Tool Functions

    def get_f_rho(self, rho):
        '''
        Given the distance rho from image center, return the f(rho) function output, the z coordinate in 3d.

        Input:
            rho (float) : distance from image center
        Output:
            zp (float) : z coordinate in 3d
        '''
        zp = self.ss[0]
        r_i = 1

        for i in range(1, len(self.ss)):
            r_i *= rho
            zp += r_i * self.ss[i]
        return zp

    def get_ray_angle(self, rho):
        '''
        Given the distance rho from image center, return the angle of optical ray in degree

        Input:
            rho (float) : distance from image center
        Output:
            (float) : angle of optical ray (deg)
        '''
        return np.rad2deg(np.arctan2(self.get_f_rho(rho), rho))

    def cam2world(self, point2D):
        '''
        Projects 2D point onto unit sphere

        Input:
            point2D (list[float]) : [x, y] coordinate of 2d points
        Output:
            point3D (list[float]) : [x, y, z] coordinate of 3d points
        '''
        invdet = 1 / (self.c - self.d * self.e)
        xp = invdet*((point2D[0] - self.xc) -
                     self.d*(point2D[1] - self.yc))
        yp = invdet*(-self.e*(point2D[0] - self.xc) +
                     self.c*(point2D[1] - self.yc))
        # distance [pixels] of  the point from the image center
        r = np.sqrt(xp*xp + yp*yp)
        zp = self.get_f_rho(r)

        point3D = [0, 0, 0]

        # normalize to unit norm
        invnorm = 1 / np.sqrt(xp*xp + yp*yp + zp*zp)
        # project onto unit sphere
        point3D[0] = invnorm*xp
        point3D[1] = invnorm*yp
        point3D[2] = invnorm*zp

        return point3D

    def world2cam(self, point3D):
        '''
        Projects 3D point into the image pixel

        Input:
            point3D (list[float]) : [x, y, z] coordinate of 3d points
        Output:
            u (float) : column pixel coordinate along width
            v (float) : row pixel coordinate along height
        '''
        invpol = self.invpol
        length_invpol = len(invpol)

        norm = np.sqrt(point3D[0]*point3D[0] + point3D[1]*point3D[1])
        theta = np.arctan2(point3D[2], norm)

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

            v = x * self.c + y * self.d + self.yc
            u = x * self.e + y + self.xc

        else:

            v = self.yc
            u = self.xc

        return u, v

    # Projection Functions

    def create_LUT_warp(self, Rmax, Rmin, height, width):
        '''
        Deprecated
        Create look up table to warp panorama image into ring image
        '''
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

    def create_LUT_cuboid(self, FOV_angle=90):
        '''
        Create Look Up Table (LUT) for remapping omnidirectional image into cuboid panorama image

        Input:
            None
        Output:
            maps (list[list[numpy.ndarray]]) : list of [mapx, mapy] in 4 directions and front left, front right
        '''

        elevation_angle = self.get_ray_angle(self.Rmax)
        depression_angle = self.get_ray_angle(self.Rmin)

        height = self.Rmax - self.Rmin

        # R = distance from image plane to the optical center
        # R = h * tan(theta1) + h * tan(theta2)
        # width = R * tan(FOV/2) * 2
        R = height / \
            (np.tan(np.deg2rad(elevation_angle)) +
             np.tan(np.deg2rad(-depression_angle)))
        width = R * np.tan(np.deg2rad(FOV_angle / 2)) * 2

        z_offset = R * np.tan(np.deg2rad(elevation_angle))

        height = np.int32(height)
        width = np.int32(width)

        mapx = np.zeros((height, width), dtype=np.float32)
        mapy = np.zeros((height, width), dtype=np.float32)

        maps = []

        #    front, left, back, right, front left,              front right
        # x:   j-w,   -R,  w-j,     R, sqrt2/2 * j - sqrt2 * R, sqrt2/2 * j
        # y:     R,  j-w,   -R,   w-j, sqrt2/2 * j            , sqrt2 * R - sqrt2 / 2 * j
        #
        def get_project_coord(index, j, width, R):
            w = width / 2
            sqrt2 = np.math.sqrt(2)
            if index == 0:
                return (j-w, R)
            elif index == 1:
                return (-R, j-w)
            elif index == 2:
                return (w-j, -R)
            elif index == 3:
                return (R, w-j)
            elif index == 4:
                return (sqrt2 / 2 * j - sqrt2 * R, sqrt2 / 2 * j)
            elif index == 5:
                return (sqrt2 / 2 * j, sqrt2 * R - sqrt2 / 2 * j)

        for k in range(6):
            for i in range(height):
                for j in range(width):
                    # Transform the dst image coordinate to XYZ coordinate
                    x, y = get_project_coord(k, j, width, R)
                    z = z_offset - i

                    # Reproject onto image pixel coordinate
                    u, v = self.world2cam([-y, x, z])

                    mapx[i][j] = u
                    mapy[i][j] = v

            maps.append([mapx.copy(), mapy.copy()])

        self.LUT_cuboid = maps
        return maps

    def create_LUT_cylinder(self):
        '''
        Create Look Up Table (LUT) for remapping omnidirectional image into cylinder panoramic image

        Input:
            None
        Output:
            mapx (numpy.ndarray) : mapx to be used in np.remap
            mapy (numpy.ndarray) : mapy to be used in np.remap
        '''
        height = np.int32(self.Rmax - self.Rmin)
        width = np.int32(np.pi * (self.Rmax + self.Rmin))

        mapx = np.zeros((height, width), dtype=np.float32)
        mapy = np.zeros((height, width), dtype=np.float32)

        for i in range(height):
            for j in range(width):
                # Note, if you would like to flip the image, just inverte the sign of theta
                theta = -(j) / width * 2 * math.pi
                rho = self.Rmax - (self.Rmax-self.Rmin) / height * i
                mapx[i][j] = self.xc + rho * math.sin(theta)
                mapy[i][j] = self.yc + rho * math.cos(theta)
        self.LUT_cylinder = (mapx, mapy)
        return mapx, mapy

    def split_cuboid_image(self, src):
        '''
        Split a rectified cuboid image into 4 directions

        Input:
            src (numpy.ndarray) : Input cuboid panaroma image
        Output:
            (list[numpy.ndarray]) : 4 seperate images and full image [front, left, back, right, full]
        '''
        width = src.shape[1] // 4
        right = src[:, :width]
        back = src[:, width:width*2]
        left = src[:, width*2:width*3]
        front = src[:, width*3:width*4]

        return [front, left, back, right, src]

    def cylinder_rectify(self, src):
        '''
        Rectify omnidirectional ring image into cylinder panorama image

        Input:
            src (numpy.ndarray) : Input omnidirectional ring image
        Output:
            img_rectified (list[numpy.ndarray]) : Rectified panorama image
        '''

        if not self.LUT_cylinder:
            # Create panoramic look up table
            self.LUT_cylinder = self.create_LUT_cylinder()

        mapx, mapy = self.LUT_cylinder

        # Remap into panoramic image
        dst = cv.remap(src, mapx, mapy, cv.INTER_LINEAR)

        return [dst]

    def cuboid_rectify(self, src):
        '''
        Rectify omnidirectional image into cuboid panorama image

        Input:
            src (numpy.ndarray) : Input omnidirectional ring image
        Output:
            imgs (list[numpy.ndarray]) : 4 seperate images and full image [front, left, back, right, full]
        '''

        if not self.LUT_cuboid:
            self.LUT_cuboid = self.create_LUT_cuboid()
        cuboid_maps = self.LUT_cuboid

        mapx_all = np.concatenate([m[0] for m in cuboid_maps[3::-1]], axis=1)
        mapy_all = np.concatenate([m[1] for m in cuboid_maps[3::-1]], axis=1)

        full = cv.remap(src, mapx_all, mapy_all,
                        cv.INTER_LINEAR, cv.BORDER_CONSTANT, 0)
        imgs = self.split_cuboid_image(full)

        return imgs

    def cuboid_rectify_plus(self, src):
        '''
        Rectify omnidirectional image into cuboid panorama image and front-left, front-right perspective image

        Input:
            src (numpy.ndarray) : Input omnidirectional ring image
        Output:
            imgs (list[numpy.ndarray]) : 6 seperate images and full image [front, left, back, right, full, front-left, front-right]
        '''

        if not self.LUT_cuboid:
            self.LUT_cuboid = self.create_LUT_cuboid()
        cuboid_maps = self.LUT_cuboid

        # 4 directions rectify, reverse the order for image stitching
        mapx_full = np.concatenate(
            [m[0] for m in cuboid_maps[3::-1]], axis=1)
        mapy_full = np.concatenate(
            [m[1] for m in cuboid_maps[3::-1]], axis=1)
        img_full = cv.remap(src, mapx_full, mapy_full,
                            cv.INTER_LINEAR, cv.BORDER_CONSTANT, 0)
        imgs = self.split_cuboid_image(img_full)

        # front-left and front-right rectify
        mapx_45 = np.concatenate([m[0] for m in cuboid_maps[4:]], axis=1)
        mapy_45 = np.concatenate([m[1] for m in cuboid_maps[4:]], axis=1)
        img_45 = cv.remap(src, mapx_45, mapy_45,
                          cv.INTER_LINEAR, cv.BORDER_CONSTANT, 0)

        width = cuboid_maps[0][0].shape[1]
        front_left = img_45[:, :width]
        front_right = img_45[:, width:width*2]

        imgs.extend([front_left, front_right])

        return imgs


class OmniUnwarp():
    '''
    User interface for unwarpping images
    '''

    def __init__(self, **kwargs):

        self.model = None  # Scaramuzza model

        # Image parameter
        self.left_cropped_pixel = 300
        self.right_copped_pixel = 320
        self.height = 1080
        self.width = 1920

        self.scara_default_param = {
            'cde': [0.999998, 3e-06, 2.6e-05],
            'invpol': [12.0,
                       443.183827,
                       308.133563,
                       28.505708,
                       27.497972,
                       26.856912,
                       9.619478,
                       1.117213,
                       2.966127,
                       3.20651,
                       2.026977,
                       0.94849,
                       0.200601],
            'shape': [1080.0, 1300.0],
            'ss': [5.0, -266.588, 0.0, 0.001220001, -5.78173e-07, 2.003631e-09],
            'xy': [545.872616, 674.318875],
            'crop_pixel': [300, 320],
            'R': [600, 200]
        }

        self.mode = kwargs.get('mode', 'cuboid')
        self.version = kwargs.get('version', '0.2.2')

        if self.read_calib_results(kwargs.get('calib_results_path', '')):
            self.model = ScaramuzzaOcamModel(param=self.calib_results)
        else:
            print("Using default value!")
            self.model = ScaramuzzaOcamModel(param=self.scara_default_param)

        mask = self.create_mask()
        self.model.create_LUT_cylinder()
        self.model.create_LUT_cuboid()
        self.model.mask_cylinder = self.model.cylinder_rectify(mask)
        self.model.mask_cuboid_plus = self.model.cuboid_rectify_plus(mask)

    def read_calib_results(self, calib_results_path):
        '''
        Read calib_results.txt from Matlab calibration
        '''

        try:
            with open(calib_results_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError as fnfe:
            print(fnfe)
            return False
        def split_by_comma(line): return np.asarray((line.strip().split()),
                                                    dtype=np.float64).tolist()
        self.calib_results = {
            'ss': split_by_comma(lines[2]),
            'invpol': split_by_comma(lines[6]),
            'xy': split_by_comma(lines[10]),
            'cde': split_by_comma(lines[14]),
            'shape': split_by_comma(lines[18]),
            'R': split_by_comma(lines[22])
        }

        pixel = split_by_comma(lines[26])
        self.left_cropped_pixel, self.right_copped_pixel = int(
            pixel[0]), int(pixel[1])
        return True

    def crop_img(self, src):
        '''
        Crop the image to select smaller region of interest.
        Current setting is specific to camera and images used in the example.

        Input:
            src (numpy.ndarray) : Input uncropped image
        Output:
            dst (numpy.ndarray) : Cropped image with only omnidirectional FOV
        '''

        # Reshape the image
        cropped = src[:, self.left_cropped_pixel:self.width -
                      self.right_copped_pixel]

        return cropped

    def create_mask(self):
        '''
        Create mask. User need to define mask 

        Output:
            mask (numpy.ndarray) : mask image
        '''
        h = self.height
        w = self.width - self.left_cropped_pixel - self.right_copped_pixel

        # Erase center oval
        mask1 = np.full((h, w), 255, dtype=np.uint8)
        center1 = (674, 546)
        cv.ellipse(mask1, center1, (240, 240), 0,
                   0, 360, (0, 0, 0), thickness=-1)

        # Preserve large circle
        mask2 = np.zeros((h, w), dtype=np.uint8)

        center2 = (674, 546)
        cv.circle(mask2, center2, 600, (255, 255, 255), thickness=-1)

        mask = cv.bitwise_and(mask1, mask2)
        return mask

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
        rot_mat = cv.getRotationMatrix2D(center, angle, 1.0)
        result = cv.warpAffine(
            src, rot_mat, src.shape[1::-1], flags=cv.INTER_LINEAR)
        return result

    def cuboid_rectify(self, src):
        '''
        Perform cuboid unwarp and return the unwarped image, mask, and label
        Version 0.2.1: Cuboid rectify, return [front, left, back, right, full]
        Version 0.2.2: Cuboid rectify plus, return [front, left, back, right, full, front-left, front-right]

        Input:
            src (numpy.ndarray) : Input image, default size is (1920, 1080)
        Output:
            imgs   (list[numpy.ndarray]) : Total 5 (7) unwarped images, including 4 seperate images and full image 
                                           [front, left, back, right, full, (front-left, front-right)]
            masks  (list[numpy.ndarray]) : Masks for corresponding imgs [front, left, back, right, full, (front-left, front-right)]
            labels (list[str])           : Labels of corresponding imgs [front, left, back, right, full, (front-left, front-right)]
        '''

        cropped = self.crop_img(src)

        if self.version == '0.2.1':
            # cuboid
            imgs = self.model.cuboid_rectify(cropped)
            masks = self.model.mask_cuboid_plus[:5]
            labels = ['front', 'left', 'back', 'right', 'full']
        elif self.version == '0.2.2':
            # cuboid + front-left + front-right
            imgs = self.model.cuboid_rectify_plus(cropped)
            masks = self.model.mask_cuboid_plus
            labels = ['front', 'left', 'back', 'right',
                      'full', 'front-left', 'front-right']
        else:
            # default using cuboid rectify plus
            imgs = self.model.cuboid_rectify_plus(cropped)
            masks = self.model.mask_cuboid_plus
            labels = ['front', 'left', 'back', 'right',
                      'full', 'front-left', 'front-right']

        return imgs, masks, labels

    def cylinder_rectify(self, src):
        '''
        Perform cylinder unwarp and return the unwarped image, mask, and label

        Input:
            src (numpy.ndarray) : Input image, default size is (1920, 1080)
        Output:
            imgs  (list[numpy.ndarray]) : 1 unwarped image [full]
            masks (list[numpy.ndarray]) : Mask for corresponding imgs [full]
            labels (list[str])          : Labels of corresponding imgs [full]
        '''
        cropped = self.crop_img(src)
        imgs = self.model.cylinder_rectify(cropped)
        masks = self.model.mask_cylinder
        labels = ['full']

        return imgs, masks, labels

    def rectify(self, src):
        '''
        Perform unwarp with corresponding mode and return the unwarped image, mask, and label
        Mode is specified in the parameters

        Input:
            src (numpy.ndarray) : Input image, default size is (1920, 1080)
        Output:
            imgs  (list[numpy.ndarray]) : 1 unwarped image [full]
            masks (list[numpy.ndarray]) : Mask for corresponding imgs [full]
            labels (list[str])          : Labels of corresponding imgs [full]
        '''
        if self.mode == 'cuboid':
            return self.cuboid_rectify(src)
        elif self.mode == 'cylinder':
            return self.cylinder_rectify(src)
        else:
            print("Please specify rectify mode")
            return [], [], []
