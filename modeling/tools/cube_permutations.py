import numpy as np 

def rot90(m, k=1, axis=2):
    """Rotate an array k*90 degrees in the counter-clockwise direction around the given axis"""
    m = np.swapaxes(m, 2, axis)
    m = np.rot90(m, k)
    m = np.swapaxes(m, 2, axis)
    return m

def rotations4(polycube, axis):
    """List the four rotations of the given cube about the given axis."""
    for i in range(4):
        yield rot90(polycube, i, axis)

def rotations24(polycube):
    # imagine shape is pointing in axis 0 (up)

    # 4 rotations about axis 0
    yield from rotations4(polycube, 0)

    # rotate 180 about axis 1, now shape is pointing down in axis 0
    # 4 rotations about axis 0
    yield from rotations4(rot90(polycube, 2, axis=1), 0)

    # rotate 90 or 270 about axis 1, now shape is pointing in axis 2
    # 8 rotations about axis 2
    yield from rotations4(rot90(polycube, axis=1), 2)
    yield from rotations4(rot90(polycube, -1, axis=1), 2)

    # rotate about axis 2, now shape is pointing in axis 1
    # 8 rotations about axis 1
    yield from rotations4(rot90(polycube, axis=2), 1)
    yield from rotations4(rot90(polycube, -1, axis=2), 1)

def permutations(cube):
    """yields all 96 permutations of a cube"""
    yield from rotations24(cube)
    yield from rotations24(np.flip(cube, 0))
    yield from rotations24(np.flip(cube, 1))
    yield from rotations24(np.flip(cube, 2))

def single_rotations4(cube, axis, idx):
    rot_idx = idx%4
    return rot90(cube, rot_idx, axis)

def single_rotations24(cube, idx):
    rot_idx = idx%6
    if rot_idx == 0:
        return single_rotations4(cube, 0, idx)
    elif rot_idx == 1:
        return single_rotations4(rot90(cube, 2, axis=1), 0, idx)
    elif rot_idx == 2:
        return single_rotations4(rot90(cube, axis=1), 2, idx)
    elif rot_idx == 3:
        return single_rotations4(rot90(cube, -1, axis=1), 2, idx)
    elif rot_idx == 4:
        return single_rotations4(rot90(cube, axis=2), 1, idx)
    elif rot_idx == 5:
        return single_rotations4(rot90(cube, -1, axis=1), 2, idx)
    return None

def single_perm(cube, idx):
    flip_idx = idx//24
    if flip_idx == 0:
        return single_rotations24(cube, idx).copy()
    else:
        return single_rotations24(np.flip(cube, flip_idx-1), idx).copy()
