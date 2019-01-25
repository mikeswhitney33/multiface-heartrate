import numpy as np

def arbitrary_pad(li):
    """pads and merges arbitrarily many 3-d arrays of equal final dimension. Expects list of arrays"""


def padded_stack(a, b, how='vert', where='+', shift=0,
                 a_transform=None, b_transform=None):
    '''Merge 2-dimensional numpy arrays with different shapes
    Parameters
    ----------
    a, b : numpy arrays
        The arrays to be merged
    how : optional string (default = 'vert')
        The method through wich the arrays should be stacked. `'Vert'`
        is analogous to `np.vstack`. `'Horiz'` maps to `np.hstack`.
    where : optional string (default = '+')
        The placement of the arrays relative to each other. Keeping in
        mind that the origin of an array's index is in the upper-left
        corner, `'+'` indicates that the second array will be placed
        at higher index relative to the first array. Essentially:
         - if how == 'vert'
            - `'+'` -> `a` is above `b`
            - `'-'` -> `a` is below `b`
         - if how == 'horiz'
            - `'+'` -> `a` is to the left of `b`
            - `'-'` -> `a` is to the right of `b`
        See the examples for more info.
    shift : int (default = 0)
        The number of indices the second array should be shifted in
        axis other than the one being merged. In other words, vertically
        stacked arrays can be shifted horizontally, and horizontally
        stacked arrays can be shifted vertically.
    [a|b]_transform : function, lambda, or None (default)
        Individual transformations that will be applied to the arrays
        *prior* to being merged. This can be numeric of even alter the
        shapes (e.g., `np.flipud`, `np.transpose`)
    Returns
    -------
    Stacked : numpy array
        The merged and padded array
    Examples
    --------
    >>> import pygridtools as pgt
    >>> a = np.arange(12).reshape(4, 3) * 1.0
    >>> b = np.arange(8).reshape(2, 4) * -1.0
    >>> pgt.padded_stack(a, b, how='vert', where='+', shift=1)
        array([[  0.,   1.,   2.,  nan,  nan],
               [  3.,   4.,   5.,  nan,  nan],
               [  6.,   7.,   8.,  nan,  nan],
               [  9.,  10.,  11.,  nan,  nan],
               [ nan,  -0.,  -1.,  -2.,  -3.],
               [ nan,  -4.,  -5.,  -6.,  -7.]])
    >>> pgt.padded_stack(a, b, how='h', where='-', shift=-2)
        array([[ nan,  nan,  nan,  nan,   0.,   1.,   2.],
               [ nan,  nan,  nan,  nan,   3.,   4.,   5.],
               [ -0.,  -1.,  -2.,  -3.,   6.,   7.,   8.],
               [ -4.,  -5.,  -6.,  -7.,   9.,  10.,  11.]]
    >>> pgt.padded_stack(a, b, how='h', where='-', shift=-2,
            a_transform=lambda a: np.transpose(np.flipud(a)))
        array([[ nan,  nan,  nan,  nan,   2.,   5.,   8.,  11.],
               [ nan,  nan,  nan,  nan,   1.,   4.,   7.,  10.],
               [ -0.,  -1.,  -2.,  -3.,   0.,   3.,   6.,   9.],
               [ -4.,  -5.,  -6.,  -7.,  nan,  nan,  nan,  nan]])
    '''
    a = np.asarray(a)
    b = np.asarray(b)

    if a_transform is None:
        a_transform = lambda a: a

    if b_transform is None:
        b_transform = lambda b: b

    if where == '-':
        stacked = padded_stack(
            b, a, shift=shift,
            where='+', how=how,
            a_transform=b_transform,
            b_transform=a_transform
        )
    else:
        if how.lower() in ('horizontal', 'horiz', 'h'):
            stacked = padded_stack(
                a.T, b.T, shift=shift,
                where=where, how='v',
                a_transform=a_transform,
                b_transform=b_transform
            ).T


        elif how.lower() in ('vertical', 'vert', 'v'):

            at = a_transform(a)
            bt = b_transform(b)

            a_pad_left = 0
            a_pad_right = 0
            b_pad_left = 0
            b_pad_right = 0

            diff_cols = at.shape[1] - (bt.shape[1] + shift)

            if shift > 0:
                b_pad_left = shift
            elif shift < 0:
                a_pad_left = abs(shift)

            if diff_cols > 0:
                b_pad_right = diff_cols
            else:
                a_pad_right = abs(diff_cols)

            v_pads = (0, 0)
            x_pads = (v_pads, (a_pad_left, a_pad_right))
            y_pads = (v_pads, (b_pad_left, b_pad_right))

            mode = 'constant'
            fill = (np.nan, np.nan)
            stacked = np.vstack([
                np.pad(at, x_pads, mode=mode, constant_values=fill),
                np.pad(bt, y_pads, mode=mode, constant_values=fill)
            ])

        else:
            gen_msg = 'how must be either "horizontal" or "vertical"'
            raise ValueError(gen_msg)

    return stacked