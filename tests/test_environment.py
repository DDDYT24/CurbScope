"""The pinned environment imports and works together (numpy 1.26 with every compiled package)."""

import numpy as np


def test_compiled_packages_work_with_the_pinned_numpy():
    import cv2
    import pyarrow
    import scipy.optimize
    import shapely

    assert np.__version__.startswith("1.26")
    assert pyarrow.array([1, 2]).to_pylist() == [1, 2]       # Streamlit tables and maps
    assert scipy.optimize.linear_sum_assignment(np.eye(2))[0].tolist() == [0, 1]
    assert shapely.Point(0, 0).buffer(1).area > 3
    assert cv2.cvtColor(np.zeros((2, 2, 3), np.uint8), cv2.COLOR_RGB2GRAY).shape == (2, 2)


def test_streamlit_folium_imports():
    import folium
    import streamlit_folium  # noqa: F401

    assert folium.Map(location=[49.0, 8.4]).location == [49.0, 8.4]
