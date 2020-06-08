

import numpy as np


##################################################################
# symmetric over z axis
def get_spherical_coords_z(X):
    # X is N x 3
    rad = np.linalg.norm(X, axis=1)
    # Inclination
    theta = np.arccos(X[:, 2] / rad)
    # Azimuth
    phi = np.arctan2(X[:, 1], X[:, 0])

    # Normalize both to be between [-1, 1]
    vv = (theta / np.pi) * 2 - 1
    uu = ((phi + np.pi) / (2 * np.pi)) * 2 - 1
    # Return N x 2
    return np.stack([uu, vv], 1)


# symmetric over x axis
def get_spherical_coords_x(X):
    # X is N x 3
    rad = np.linalg.norm(X, axis=1)
    # Inclination
    # y == 1
    # cos = 0
    # y == -1
    # cos = pi
    theta = np.arccos(X[:, 0] / rad)
    # Azimuth
    phi = np.arctan2(X[:, 2], X[:, 1])

    # Normalize both to be between [-1, 1]
    uu = (theta / np.pi) * 2 - 1
    vv = ((phi + np.pi) / (2 * np.pi)) * 2 - 1
    # Return N x 2
    return np.stack([uu, vv], 1)


#########################################################################
if __name__ == '__main__':
    
    from utils.utils_mesh import loadobj, savemeshtes
    import cv2
    
    p, f = loadobj('2.obj')
    uv = get_spherical_coords_x(p)
    uv[:, 0] = -uv[:, 0]
    
    uv[:, 1] = -uv[:, 1]
    uv = (uv + 1) / 2
    savemeshtes(p, uv, f, './2_x.obj')
    
    tex = np.zeros(shape=(256, 512, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 200)
    fontScale = 5
    fontColor = (0, 255, 255)
    lineType = 2

    cv2.putText(tex, 'Hello World!',
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    cv2.imshow('', tex)
    cv2.waitKey()
    cv2.imwrite('2_x.png', np.transpose(tex, [1, 0, 2]))

