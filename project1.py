import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from google.cloud import vision
import io
from skimage import transform


def align_images(file_path):

    # Read 8-bit color image.
    # This is an image in which the three channels are
    # concatenated vertically.

    im = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # edge detection
    row_top, rt = 0, 0
    row_down = 0
    col_top = 0
    col_down = 0
    # if im.shape[0] > 341 and im.shape[1] > 400:
    #     im = cv2.resize(im, (341, 398), interpolation=cv2.INTER_LINEAR)
    (row, col) = im.shape
    for r in range(0, row):
        if im.sum(axis=1)[r] < 200 * col and rt == 0:
            row_top = r
            rt = 1
            break
    for r in range(row - 1, 0, -1):
        if im.sum(axis=1)[r] < 200 * col:
            row_down = r
            break

    for c in range(0, col):
        if im.sum(axis=0)[c] < 200 * row:
            col_top = c
            break
    for c in range(col - 1, 0, -1):
        if im.sum(axis=0)[c] < 200 * row:
            col_down = c
            break
    im = im[row_top:row_down, col_top:col_down]

    # Find the width and height of the color image
    sz = im.shape
    height = int(sz[0] / 3)
    width = sz[1]

    # Extract the three channels from the gray scale image
    # and merge the three channels into one color image
    im_color = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(0, 3):
        im_color[:, :, i] = im[i * height:(i + 1) * height, :]

    # Allocate space for aligned image
    im_aligned = np.zeros((height, width, 3), dtype=np.uint8)

    # The blue and green channels will be aligned to the red channel.
    # So copy the red channel
    im_aligned[:, :, 2] = im_color[:, :, 2]

    # Define motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY

    # Set the warp matrix to identity.
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Set the stopping criteria for the algorithm.
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

    # Warp the blue and green channels to the red channel
    error = False
    for i in range(0, 2):
        (cc, warp_matrix) = cv2.findTransformECC(get_gradient(im_color[:, :, 2]),
            get_gradient(im_color[:, :, i]),
            warp_matrix, warp_mode, criteria, None, gaussFiltSize=5)

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use Perspective warp when the transformation is a Homography
            im_aligned[:, :, i] = cv2.warpPerspective(im_color[:, :, i], warp_matrix, (width, height),
                                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use Affine warp when the transformation is not a Homography
            im_aligned[:, :, i] = cv2.warpAffine(im_color[:, :, i], warp_matrix, (width, height),
                                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        # print(warp_matrix)

    # Show final output
    # cv2.imshow("Color Image", im_color)
    # cv2.imshow("Aligned Image", im_aligned)
    # cv2.waitKey(0)
    b, g, r = cv2.split(im_aligned)
    im_aligned = cv2.merge([r, g, b])

    # denoise
    # result = cv2.fastNlMeansDenoisingColored(im_aligned, None, 15, 15, 10, 30)

    # # contrast
    # im_aligned = cv2.normalize(result, dst=None, alpha=50, beta=10, norm_type=cv2.NORM_MINMAX)

    print(im_aligned.shape)
    return im_color, im_aligned


def get_gradient(im):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


def white_balance_1(img):
    '''
    White equilibrium method for the mean value
    :param img: cv2.imread read the image data
    :return: Return the white balance result picture data
    '''
    # 读取图像
    r, g, b = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
    # 求各个通道所占增益
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img = cv2.merge([b, g, r])
    return balance_img


def white_balance_2(img_input):
    '''
    Perfect reflection white balance
    STEP 1：Calculate the sum of R\G\B for each pixel
    STEP 2：According to the size of R+G+B value, calculate the threshold T of its former Ratio% value as the reference point
    STEP 3：For each point in the image, calculate the average cumulative sum of the R\G\B components of all points whose R+G+B value is greater than T
    STEP 4：For each point, the pixel is quantized to between [0,255]
    It is not good for images that rely on ratio and are not white in the brightest area
    :param img: cv2.imread read the image data
    :return: Return the white balance result picture data
    '''
    img = img_input.copy()
    b, g, r = cv2.split(img)
    m, n, t = img.shape
    sum_ = np.zeros(b.shape)
    for i in range(m):
        for j in range(n):
            sum_[i][j] = int(b[i][j]) + int(g[i][j]) + int(r[i][j])
    hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
    Y = 765
    num, key = 0, 0
    ratio = 0.01
    while Y >= 0:
        num += hists[Y]
        if num > m * n * ratio / 100:
            key = Y
            break
        Y = Y - 1

    sum_b, sum_g, sum_r = 0, 0, 0
    time = 0
    for i in range(m):
        for j in range(n):
            if sum_[i][j] >= key:
                sum_b += b[i][j]
                sum_g += g[i][j]
                sum_r += r[i][j]
                time = time + 1

    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time

    maxvalue = float(np.max(img))
    # maxvalue = 255
    for i in range(m):
        for j in range(n):
            b = int(img[i][j][0]) * maxvalue / int(avg_b)
            g = int(img[i][j][1]) * maxvalue / int(avg_g)
            r = int(img[i][j][2]) * maxvalue / int(avg_r)
            if b > 255:
                b = 255
            if b < 0:
                b = 0
            if g > 255:
                g = 255
            if g < 0:
                g = 0
            if r > 255:
                r = 255
            if r < 0:
                r = 0
            img[i][j][0] = b
            img[i][j][1] = g
            img[i][j][2] = r

    return img


def white_balance_3(img):
    '''
    灰度世界假设
    :param img: cv2.imread read the image data
    :return: Return the white balance result picture data
    '''
    B, G, R = np.double(img[:, :, 0]), np.double(img[:, :, 1]), np.double(img[:, :, 2])
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
    K = (B_ave + G_ave + R_ave) / 3
    Kb, Kg, Kr = K / B_ave, K / G_ave, K / R_ave
    Ba = (B * Kb)
    Ga = (G * Kg)
    Ra = (R * Kr)

    for i in range(len(Ba)):
        for j in range(len(Ba[0])):
            Ba[i][j] = 255 if Ba[i][j] > 255 else Ba[i][j]
            Ga[i][j] = 255 if Ga[i][j] > 255 else Ga[i][j]
            Ra[i][j] = 255 if Ra[i][j] > 255 else Ra[i][j]

    # print(np.mean(Ba), np.mean(Ga), np.mean(Ra))
    dst_img = np.uint8(np.zeros_like(img))
    dst_img[:, :, 0] = Ba
    dst_img[:, :, 1] = Ga
    dst_img[:, :, 2] = Ra
    return dst_img


def white_balance_4(img):
    '''
    Color offset detection and color correction method based on image analysis
    :param img: cv2.imread read the image
    :return: Return the white balance result picture data
    '''

    def detection(img):
        '''calculate Partial color value'''
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img_lab)
        d_a, d_b, M_a, M_b = 0, 0, 0, 0
        for i in range(m):
            for j in range(n):
                d_a = d_a + a[i][j]
                d_b = d_b + b[i][j]
        d_a, d_b = (d_a / (m * n)) - 128, (d_b / (n * m)) - 128
        D = np.sqrt((np.square(d_a) + np.square(d_b)))

        for i in range(m):
            for j in range(n):
                M_a = np.abs(a[i][j] - d_a - 128) + M_a
                M_b = np.abs(b[i][j] - d_b - 128) + M_b

        M_a, M_b = M_a / (m * n), M_b / (m * n)
        M = np.sqrt((np.square(M_a) + np.square(M_b)))
        k = D / M
        print('Partial color value:%f' % k)
        return

    b, g, r = cv2.split(img)
    # print(img.shape)
    m, n = b.shape
    # detection(img)

    I_r_2 = np.zeros(r.shape)
    I_b_2 = np.zeros(b.shape)
    sum_I_r_2, sum_I_r, sum_I_b_2, sum_I_b, sum_I_g = 0, 0, 0, 0, 0
    max_I_r_2, max_I_r, max_I_b_2, max_I_b, max_I_g = int(r[0][0] ** 2), int(r[0][0]), int(b[0][0] ** 2), int(
        b[0][0]), int(g[0][0])
    for i in range(m):
        for j in range(n):
            I_r_2[i][j] = int(r[i][j] ** 2)
            I_b_2[i][j] = int(b[i][j] ** 2)
            sum_I_r_2 = I_r_2[i][j] + sum_I_r_2
            sum_I_b_2 = I_b_2[i][j] + sum_I_b_2
            sum_I_g = g[i][j] + sum_I_g
            sum_I_r = r[i][j] + sum_I_r
            sum_I_b = b[i][j] + sum_I_b
            if max_I_r < r[i][j]:
                max_I_r = r[i][j]
            if max_I_r_2 < I_r_2[i][j]:
                max_I_r_2 = I_r_2[i][j]
            if max_I_g < g[i][j]:
                max_I_g = g[i][j]
            if max_I_b_2 < I_b_2[i][j]:
                max_I_b_2 = I_b_2[i][j]
            if max_I_b < b[i][j]:
                max_I_b = b[i][j]

    [u_b, v_b] = np.matmul(np.linalg.inv([[sum_I_b_2, sum_I_b], [max_I_b_2, max_I_b]]), [sum_I_g, max_I_g])
    [u_r, v_r] = np.matmul(np.linalg.inv([[sum_I_r_2, sum_I_r], [max_I_r_2, max_I_r]]), [sum_I_g, max_I_g])
    # print(u_b, v_b, u_r, v_r)
    b0, g0, r0 = np.zeros(b.shape, np.uint8), np.zeros(g.shape, np.uint8), np.zeros(r.shape, np.uint8)
    for i in range(m):
        for j in range(n):
            b_point = u_b * (b[i][j] ** 2) + v_b * b[i][j]
            g0[i][j] = g[i][j]
            # r0[i][j] = r[i][j]
            r_point = u_r * (r[i][j] ** 2) + v_r * r[i][j]
            if r_point > 255:
                r0[i][j] = 255
            else:
                if r_point < 0:
                    r0[i][j] = 0
                else:
                    r0[i][j] = r_point
            if b_point > 255:
                b0[i][j] = 255
            else:
                if b_point < 0:
                    b0[i][j] = 0
                else:
                    b0[i][j] = b_point
    return cv2.merge([b0, g0, r0])


def white_balance_5(img):
    '''
    Dynamic threshold algorithm
    The algorithm is divided into two steps: white spot detection and white spot adjustment
    But the white spot detection is not the same as the perfect reflection algorithm that considers the brightest point as white, but determined by another rule
    :param img: cv2.imread read the image
    :return: Return the white balance result picture data
    '''

    b, g, r = cv2.split(img)
    """
    YUV space
    """

    def con_num(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        if x == 0:
            return 0

    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    (y, u, v) = cv2.split(yuv_img)
    # y, u, v = cv2.split(img)
    m, n = y.shape
    sum_u, sum_v = 0, 0
    max_y = np.max(y.flatten())
    # print(max_y)
    for i in range(m):
        for j in range(n):
            sum_u = sum_u + u[i][j]
            sum_v = sum_v + v[i][j]

    avl_u = sum_u / (m * n)
    avl_v = sum_v / (m * n)
    du, dv = 0, 0
    # print(avl_u, avl_v)
    for i in range(m):
        for j in range(n):
            du = du + np.abs(u[i][j] - avl_u)
            dv = dv + np.abs(v[i][j] - avl_v)

    avl_du = du / (m * n)
    avl_dv = dv / (m * n)
    num_y, yhistogram, ysum = np.zeros(y.shape), np.zeros(256), 0
    radio = 0.5  # If the value is too large or too small, the color temperature develops to extremes
    for i in range(m):
        for j in range(n):
            value = 0
            if np.abs(u[i][j] - (avl_u + avl_du * con_num(avl_u))) < radio * avl_du or np.abs(
                    v[i][j] - (avl_v + avl_dv * con_num(avl_v))) < radio * avl_dv:
                value = 1
            else:
                value = 0

            if value <= 0:
                continue
            num_y[i][j] = y[i][j]
            yhistogram[int(num_y[i][j])] = 1 + yhistogram[int(num_y[i][j])]
            ysum += 1
    # print(yhistogram.shape)
    sum_yhistogram = 0
    # hists2, bins = np.histogram(yhistogram, 256, [0, 256])
    # print(hists2)
    Y = 255
    num, key = 0, 0
    while Y >= 0:
        num += yhistogram[Y]
        if num > 0.1 * ysum:
            # The first 10% of the bright spot is taken as the calculated value. If the value is too large,
            # it is easy to overexpose, and the adjustment range is small if the value is too small
            key = Y
            break
        Y = Y - 1
    # print(key)
    sum_r, sum_g, sum_b, num_rgb = 0, 0, 0, 0
    for i in range(m):
        for j in range(n):
            if num_y[i][j] > key:
                sum_r = sum_r + r[i][j]
                sum_g = sum_g + g[i][j]
                sum_b = sum_b + b[i][j]
                num_rgb += 1

    avl_r = sum_r / num_rgb
    avl_g = sum_g / num_rgb
    avl_b = sum_b / num_rgb

    for i in range(m):
        for j in range(n):
            b_point = int(b[i][j]) * int(max_y) / avl_b
            g_point = int(g[i][j]) * int(max_y) / avl_g
            r_point = int(r[i][j]) * int(max_y) / avl_r
            if b_point > 255:
                b[i][j] = 255
            else:
                if b_point < 0:
                    b[i][j] = 0
                else:
                    b[i][j] = b_point
            if g_point > 255:
                g[i][j] = 255
            else:
                if g_point < 0:
                    g[i][j] = 0
                else:
                    g[i][j] = g_point
            if r_point > 255:
                r[i][j] = 255
            else:
                if r_point < 0:
                    r[i][j] = 0
                else:
                    r[i][j] = r_point

    return cv2.merge([b, g, r])


def detect_faces(path):
    """Detects faces in an image."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    print('Faces:')

    for face in faces:
        print('anger: {}'.format(likelihood_name[face.anger_likelihood]))
        print('joy: {}'.format(likelihood_name[face.joy_likelihood]))
        print('surprise: {}'.format(likelihood_name[face.surprise_likelihood]))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in face.bounding_poly.vertices])

        print('face bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return vertices


def warp(points_of_interest, projection, image):
    """

    :param points_of_interest: vertices in the face detection
    :param projection: set by yourself
    :param image: image data
    :return:
    """
    points_of_interest = np.array(points_of_interest)
    projection = np.array(projection)
    tform = transform.estimate_transform('projective', points_of_interest, projection)
    tf_img_warp = transform.warp(image, tform.inverse, mode='edge')
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    fig, ax = plt.subplots(1, 2, figsize=(15, 10), dpi=80)
    ax[0].set_title(f'Original', fontsize=15)
    ax[0].imshow(image)
    ax[0].set_axis_off();
    ax[1].set_title(f'Transformed', fontsize=15)
    ax[1].imshow(tf_img_warp)
    ax[1].set_axis_off()


if __name__ == '__main__':
    for file in os.listdir('data'):
        file_path = 'data/{}'.format(file)
        print(file_path)
        color_image, align_image = align_images(file_path)

        # enhance contrast if needed
        img_norm = cv2.normalize(align_image, dst=None, alpha=350, beta=10, norm_type=cv2.NORM_MINMAX)
        # transform into different color space if needed
        img_hsv = cv2.cvtColor(align_image, cv2.COLOR_BGR2HSV)
        img_ycrcb = cv2.cvtColor(align_image, cv2.COLOR_BGR2YCrCb)

        plt.imshow(align_image)
        plt.show()
        cv2.imwrite('align_images/{}'.format(file), align_image)
