import os
import cv2
import numpy as np

"""Github YonV1943
Mouse detection for SIAT
"""

"""utils"""


def auto_canny(image, sigma=0.33):
    """ automatically adjust the two params of canny edge detection
    source: Zero-parameter, automatic Canny edge detection with Python and OpenCV 2015-04-06
    modify: Github YonV1943
    """
    v = np.median(image)
    lower = max(0, int((1.0 - sigma) * v))
    upper = min(255, int((1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def show_video_file(video_path):
    # cap_path = "D:\\Download\\ymaze-pmq\\14# 00_00_03.20-00_08_03.20.avi"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        is_opened, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.imwrite('test04.png', frame)
        cv2.waitKey(20)
    cap.release()


def expand_grey_to_rgb(image):
    return np.repeat(image[:, :, np.newaxis], axis=2, repeats=3)


def draw_line__polar_coord(image, rho, theta, thickness=0):
    """
    thickness=0 means don't draw
    """
    line_len = image.shape[1]

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 - line_len * b)
    x2 = int(x0 + line_len * b)
    y1 = int(y0 + line_len * a)
    y2 = int(y0 - line_len * a)

    if thickness != 0:
        cv2.line(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=thickness)
    # dis_norm = rho * (np.pi / line_len)
    return x1, y1, x2, y2


def line_intersection(line1, line2):
    """print line_intersection((A, B), (C, D))
    line = ((x1, y1), (x2, y2))

    How do I compute the intersection point of two lines?
    source: https://stackoverflow.com/a/20677983/9293137
    modify: Github YonV1943
    """
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(x_diff, y_diff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div
    return x, y


def hex_str_to_rgb_tuple(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))


def draw_3area(img, the_3lines, mid_point, width, length):
    # sort lines according theta
    the_3thetas_sort = the_3lines[:, 1].argsort()
    the_3lines = the_3lines[the_3thetas_sort]

    mask1 = np.zeros(img.shape[:2])
    mask2 = np.zeros(img.shape[:2])
    mask1_3 = np.zeros((3, *img.shape[:2]), dtype=np.int)
    mask2_3 = np.zeros((3, *img.shape[:2]), dtype=np.int)
    for i, line in enumerate(the_3lines):
        theta = line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0, y0 = mid_point
        x1 = int(x0 - length * b)
        y1 = int(y0 + length * a)
        x2 = int(x0 + length * b)
        y2 = int(y0 - length * a)

        if i == 1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        cv2.line(mask1, (x0, y0), (x1, y1), color=1, thickness=width)
        cv2.line(mask2, (x0, y0), (x2, y2), color=1, thickness=width)

        cv2.line(mask1_3[i, :, :], (x0, y0), (x1, y1), color=1, thickness=width)
        cv2.line(mask2_3[i, :, :], (x0, y0), (x2, y2), color=1, thickness=width)

    avg_pixel1 = (img * mask1[:, :, np.newaxis]).sum() / mask1.sum()
    avg_pixel2 = (img * mask2[:, :, np.newaxis]).sum() / mask2.sum()

    mask_out = mask1 if avg_pixel1 > avg_pixel2 else mask2
    mask3_out = mask1_3 if avg_pixel1 > avg_pixel2 else mask2_3
    return mask_out, mask3_out


"""methods"""


def convert_mask3_to_alpha(mask3):
    mask = mask3.sum(axis=0)
    mask = np.minimum(mask, 1)
    mask = mask[:, :, np.newaxis].astype(np.float32)
    mask[mask == 0] = 0.5
    return mask


def find_3pairs_lines(lines):
    thetas = np.array([item[1] for item in lines])  # item = (dis_norm, theta, x1, y1, x2, y2)
    thresh = np.pi / 90  # np.pi / 180
    find0 = find1 = find2 = None
    print(thetas.round(3))
    print(';')
    mod_num = np.pi
    for theta in thetas:
        diff = thetas - theta
        find0 = (diff - np.pi / 3 * 0) % mod_num
        find0 = np.logical_or(find0 < thresh, find0 > mod_num - thresh)
        find1 = (diff - np.pi / 3 * 1) % mod_num
        find1 = np.logical_or(find1 < thresh, find1 > mod_num - thresh)
        find2 = (diff - np.pi / 3 * 2) % mod_num
        find2 = np.logical_or(find2 < thresh, find2 > mod_num - thresh)

        if find0.sum() >= 2 and find1.sum() >= 2 and find2.sum() >= 2:
            break
    lines = np.array(lines)

    res = [lines[i] for i in (find0, find1, find2)]
    widths = list()
    for the_2lines in res:
        if len(the_2lines) == 2:
            width = the_2lines[0][0] - the_2lines[1][0]
            widths.append(width)
    width = np.mean(np.abs(widths))
    for i, the_2lines in enumerate(res):
        if len(the_2lines) > 2:
            rhos = np.array([line[0] for line in the_2lines])
            print(width, rhos)
            for rho in rhos:
                diff = np.abs(rhos - rho)
                find0 = diff == 0
                find1 = np.logical_and(width * 0.8 < diff, diff < width * 1.2)

                if find0.sum() > 0 and find1.sum() > 0:
                    break

            find_01 = np.logical_or(find0, find1)
            res[i] = the_2lines[find_01]

    return res


def find_mouse__cnt_pnt(img, mask, ):
    mask_img = img[:, :, 0] * mask + (1 - mask) * 255
    mask_img = mask_img.astype(np.uint8)
    mask_img = cv2.blur(mask_img, (5, 5))
    thresh = 255 - cv2.threshold(mask_img, 96, 255, cv2.THRESH_BINARY)[1]
    # thresh = cv2.adaptiveThreshold(mask_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # thresh = np.repeat(thresh[:, :, np.newaxis], axis=2, repeats=3)

    tmp = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = tmp[-2:]
    # thresh, contours, hierarchy = tmp

    longest_len = 0
    longest_cnt = None
    for cnt in contours:
        if cnt.shape[0] > longest_len:
            longest_len = cnt.shape[0]
            longest_cnt = cnt
    cnt_array = np.array(longest_cnt)
    cnt_array = cnt_array.reshape((-1, 2))
    mouse_pnt = cnt_array.mean(axis=0).astype(np.int)
    mouse_pnt = tuple(mouse_pnt)
    return contours, mouse_pnt


def where_is_the_mouse(mask3, pnt):
    y, x = pnt
    where_list = [*mask3[:, x, y], 1]
    # print(where_list)

    if sum(where_list) > 2:
        area_id = 3  # 3 means in middle area
    else:
        area_id = where_list.index(1)  # {0， 1， 2} means area_id
    # 4 means not in interesting area.
    return area_id


"""run"""


def get_interest_mask(img='./test01.png'):
    cv2.namedWindow('', cv2.WINDOW_GUI_NORMAL)
    wait_time = 234
    max_line_num = 64

    if isinstance(img, str):
        img_path = img
        img = cv2.imread(img_path)
    img = cv2.blur(img, (3, 3))
    img_h, img_w, img_c = img.shape
    cv2.imshow('', img)
    cv2.waitKey(wait_time)

    edges = auto_canny(img)
    print('; detect edge')
    show = expand_grey_to_rgb(edges)
    cv2.imshow('', show)
    cv2.waitKey(wait_time)

    """cv2.HoughLines()
    rho: the pixel width of the line
    theta: the angle of the line
    threshold: the total number of the line
    """
    lines0 = cv2.HoughLines(edges, rho=1, theta=np.pi / 360, threshold=img_h // 4)
    print('; detect lines by using cv2.HoughLines')

    lines1 = list()  # rho, theta
    for line in lines0[:max_line_num]:
        rho, theta = line[0]
        lines1.append((rho, theta))
    print(f'; choose top {max_line_num} lines')

    show = img.copy()
    for rho, theta in lines1:
        draw_line__polar_coord(show, rho, theta, thickness=3)
    print('; draw these lines')
    cv2.imshow('', show)
    cv2.waitKey(wait_time)

    lines2 = list()
    min_distance = 0.1 ** 2
    rho_norm_k = np.pi / img_w
    for rho1, theta1 in lines1:
        rho_norm1 = rho1 * rho_norm_k
        is_add = True
        for rho2, theta2 in lines2:
            rho_norm2 = rho2 * rho_norm_k
            distance = (rho_norm2 - rho_norm1) ** 2 + (theta2 - theta1) ** 2
            if distance < min_distance:
                is_add = False
                break
        if is_add:
            lines2.append((rho1, theta1))
    print(f'; remove the similar lines, from {len(lines1)} to {len(lines2)}')

    lines3 = list()  # rho, theta, x1, y1, x2, y2
    show = img.copy()
    for rho, theta in lines2:
        x1, y1, x2, y2 = draw_line__polar_coord(show, rho, theta, thickness=3)
        lines3.append((rho, theta, x1, y1, x2, y2))
    print('; convert lines from polar coord into Cartesian coord')
    cv2.imshow('', show)
    cv2.waitKey(wait_time)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    the_3pairs_lines = find_3pairs_lines(lines3)
    show = img.copy()
    for i, the_2lines in enumerate(the_3pairs_lines):
        for line in the_2lines:
            rho, _theta, x1, y1, x2, y2 = [int(n) for n in line]
            rgb = hex_str_to_rgb_tuple(colors[i % 10])
            bgr = [rgb[i] for i in (2, 1, 0)]
            cv2.line(show, (x1, y1), (x2, y2), color=bgr, thickness=3)
    print('; draw 3 pairs lines')
    cv2.imshow('', show)
    cv2.waitKey(wait_time)

    the_3lines = np.mean(the_3pairs_lines, axis=1)
    show = img.copy()
    for i, line in enumerate(the_3lines):
        rho, _theta, x1, y1, x2, y2 = [int(n) for n in line]
        rgb = hex_str_to_rgb_tuple(colors[i % 10])
        bgr = [rgb[i] for i in (2, 1, 0)]
        cv2.line(show, (x1, y1), (x2, y2), color=bgr, thickness=3)
    print('; draw 3 pairs lines')
    cv2.imshow('', show)
    cv2.waitKey(wait_time)

    the_4p_3lines = [((item[2], item[3]), (item[4], item[5])) for item in the_3lines]
    three_point = [line_intersection(the_4p_3lines[0], the_4p_3lines[1]),
                   line_intersection(the_4p_3lines[0], the_4p_3lines[2]),
                   line_intersection(the_4p_3lines[1], the_4p_3lines[2]), ]
    mid_point = np.mean(three_point, axis=0).astype(np.int)
    print(f'; find the mid_point: {mid_point}')

    widths = list()
    for lines in the_3pairs_lines:
        width = abs(lines[0][0] - lines[1][0])
        widths.append(width)
    width = int(np.mean(widths))
    length = int(width * 5.2)
    print(f'; estimate the passageway: width is {width}, length is {length}: ')

    mask_out, mask3_out = draw_3area(img, the_3lines, mid_point, width, length)
    show = mask_out
    cv2.imshow('', show)
    cv2.waitKey(wait_time)
    print('; find the interesting regions')

    show = mask3_out.transpose((1, 2, 0))
    show = (show * 255).astype(np.uint8)
    cv2.imshow('', show)
    cv2.waitKey(wait_time)
    print('; find 3 interesting regions')

    mask3 = mask3_out.sum(axis=0)
    mask3 = mask3[:, :, np.newaxis].astype(np.float32)
    mask3[mask3 == 0] = 0.5

    show = img.copy() * mask3
    show = show.astype(np.uint8)
    cv2.imshow('', show)
    cv2.waitKey(wait_time)
    print('; highlight interesting regions')

    mouse_cnt, mouse_pnt = find_mouse__cnt_pnt(img, mask_out)
    show = cv2.polylines(show, mouse_cnt, True, (0, 255, 0), 2)
    cv2.imshow('', show)
    cv2.waitKey(wait_time)

    cv2.circle(show, mouse_pnt, width, color=(0, 255, 255), thickness=3)
    cv2.imshow('', show)
    cv2.waitKey(wait_time)
    print(f'; mouse coordinates: {mouse_pnt}')

    return mask_out, mask3_out, width


def run__pipeline(video_path):
    print(';;;;;', video_path)
    # video_path = "D:\\Download\\ymaze-pmq\\14# 00_00_03.20-00_08_03.20.avi"
    video_cap = cv2.VideoCapture(video_path)

    cv2.namedWindow('', cv2.WINDOW_GUI_NORMAL)

    """get the first frame"""
    is_opened, frame = video_cap.read()
    mask, mask3, width = get_interest_mask(img=frame)
    mask_alpha = convert_mask3_to_alpha(mask3)

    # show = frame.copy() * mask_alpha
    # show = show.astype(np.uint8)
    # cv2.imshow('', show)
    # cv2.waitKey(4321)
    track_mask = np.zeros(frame.shape[:2], dtype=np.float32)
    mouse_cnt, mouse_pnt = find_mouse__cnt_pnt(frame, mask)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)
    font_thick = 2

    save_list = list()

    # rate = video_cap.get(5)  # 获取帧率
    total_frame = int(video_cap.get(7))  # 获取帧数

    # while is_opened:
    import tqdm
    for i in tqdm.tqdm(range(total_frame)):
        # if i == 123:
        #     break

        try:
            mouse_cnt, mouse_pnt = find_mouse__cnt_pnt(frame, mask)
        except ValueError:
            # print('Miss one frame. Not a big problem.')
            pass
        area = where_is_the_mouse(mask3, mouse_pnt)  # area in {0, 1, 2, 3}. 3 means not in any area.
        save_list.append((mouse_pnt[0], mouse_pnt[1], area))

        # cv2.circle(track_mask, mouse_pnt, width // 3, color=1, thickness=cv2.FILLED)
        # track_mask *= 0.96

        # show = frame.copy()
        # show = cv2.polylines(show, mouse_cnt, True, (0, 255,), 2)

        # print(f'MouseCoord {mouse_pnt} Area {area}')
        # show = show * mask_alpha
        # show[:, :, 2] *= (1.0 - track_mask)
        # # show[:, :, 0] *= (1.0 - track_mask)
        #
        # cv2.putText(show, f'MouseCoord {mouse_pnt} Area {area}', (32, 32),
        #             font_face, font_scale, font_color, font_thick)
        #
        # show = show.astype(np.uint8)
        # cv2.imshow('', show)
        # cv2.waitKey(1)
        #
        # is_opened, frame = video_cap.read()

    video_cap.release()

    save_ary = np.array(save_list, dtype=np.float)
    save_name = f'mouse_points_{file_avi}.csv'
    np.savetxt(save_name, save_ary, fmt='%.0f')
    print(f'Save in :{save_name}')


if __name__ == '__main__':
    for file_avi in [n for n in os.listdir('.') if n[-4:] == '.avi']:
        run__pipeline(file_avi)
    # run__pipeline(video_path)
    # get_interest_mask(img='test05.jpg')
