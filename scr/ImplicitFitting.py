#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Li's Implicit Fitting Algorithm using Radical Basis Function with
Ellipsoid Constraint.
"""

import numpy as np
import scipy as sci
import scipy.linalg
import scipy.io as spio

import vtk
from vtk.util import numpy_support

contour_manual = 0.01
color_diffuse = [1.0, 1.0, 0.0]
sample_dims = [100] * 3

# Smoothness for visualisation
smooth_iteration = 100
smooth_factor = 0.1
smooth_angle = 60

# Depth peeling
max_peels = 100
occlusion = 0.1

# Initialisation
render_window = vtk.vtkRenderWindow()
iren = vtk.vtkRenderWindowInteractor()
renderer = vtk.vtkRenderer()


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def vtk_show(_renderer, window_name='VTK Show Window',
             width=640, height=480, has_picker=False):
    """
    Show the vtkRenderer in an vtkRenderWindow
    Only support ONE vtkRenderer
    :return: No return value
    """
    # render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(_renderer)
    render_window.SetSize(width, height)
    render_window.Render()
    # It works only after Render() is called
    render_window.SetWindowName(window_name)

    # iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(render_window)

    interactor_style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(interactor_style)

    # Add an Annoted Cube with Arrows
    cube = vtk.vtkAnnotatedCubeActor()
    cube.SetXPlusFaceText('R')
    cube.SetXMinusFaceText('L')
    cube.SetYPlusFaceText('A')
    cube.SetYMinusFaceText('P')
    cube.SetZPlusFaceText('I')
    cube.SetZMinusFaceText('S')
    cube.SetXFaceTextRotation(180)
    cube.SetYFaceTextRotation(180)
    cube.SetZFaceTextRotation(-90)
    cube.SetFaceTextScale(0.65)
    cube.GetCubeProperty().SetColor(0.5, 1.0, 1.0)
    cube.GetTextEdgesProperty().SetLineWidth(1)
    cube.GetTextEdgesProperty().SetColor(0.18, 0.28, 0.23)
    cube.GetTextEdgesProperty().SetDiffuse(0)
    cube.GetTextEdgesProperty().SetAmbient(1)

    cube.GetXPlusFaceProperty().SetColor(1, 0, 0)
    cube.GetXPlusFaceProperty().SetInterpolationToFlat()
    cube.GetXMinusFaceProperty().SetColor(1, 0, 0)
    cube.GetXMinusFaceProperty().SetInterpolationToFlat()

    cube.GetYPlusFaceProperty().SetColor(0, 1, 0)
    cube.GetYPlusFaceProperty().SetInterpolationToFlat()
    cube.GetYMinusFaceProperty().SetColor(0, 1, 0)
    cube.GetYMinusFaceProperty().SetInterpolationToFlat()

    cube.GetZPlusFaceProperty().SetColor(0, 0, 1)
    cube.GetZPlusFaceProperty().SetInterpolationToFlat()
    cube.GetZMinusFaceProperty().SetColor(0, 0, 1)
    cube.GetZMinusFaceProperty().SetInterpolationToFlat()

    text_property = vtk.vtkTextProperty()
    text_property.ItalicOn()
    text_property.ShadowOn()
    text_property.BoldOn()
    text_property.SetFontFamilyToTimes()
    text_property.SetColor(1, 0, 0)

    text_property_2 = vtk.vtkTextProperty()
    text_property_2.ShallowCopy(text_property)
    text_property_2.SetColor(0, 1, 0)
    text_property_3 = vtk.vtkTextProperty()
    text_property_3.ShallowCopy(text_property)
    text_property_3.SetColor(0, 0, 1)

    axes = vtk.vtkAxesActor()
    axes.SetShaftTypeToCylinder()
    axes.SetXAxisLabelText('X')
    axes.SetYAxisLabelText('Y')
    axes.SetZAxisLabelText('Z')
    axes.SetTotalLength(1.5, 1.5, 1.5)
    axes.GetXAxisCaptionActor2D().SetCaptionTextProperty(text_property)
    axes.GetYAxisCaptionActor2D().SetCaptionTextProperty(text_property_2)
    axes.GetZAxisCaptionActor2D().SetCaptionTextProperty(text_property_3)

    assembly = vtk.vtkPropAssembly()
    assembly.AddPart(axes)
    assembly.AddPart(cube)

    marker = vtk.vtkOrientationMarkerWidget()
    marker.SetOutlineColor(0.93, 0.57, 0.13)
    marker.SetOrientationMarker(assembly)
    marker.SetViewport(0.0, 0.0, 0.15, 0.3)
    marker.SetInteractor(iren)
    marker.EnabledOn()
    marker.InteractiveOn()

    # Add a x-y-z coordinate to the original point
    axes_coor = vtk.vtkAxes()
    axes_coor.SetOrigin(0, 0, 0)
    mapper_axes_coor = vtk.vtkPolyDataMapper()
    mapper_axes_coor.SetInputConnection(axes_coor.GetOutputPort())
    actor_axes_coor = vtk.vtkActor()
    actor_axes_coor.SetMapper(mapper_axes_coor)
    _renderer.AddActor(actor_axes_coor)

    # Add an original point and text
    add_point(_renderer, color=[0, 1, 0], radius=0.2)
    add_text(_renderer, position=[0, 0, 0], text="Origin",
             color=[0, 1, 0], scale=0.2)

    # # Create text with the x-y-z coordinate system
    # text_origin = vtk.vtkVectorText()
    # text_origin.SetText("Origin")
    # mapper_text_origin = vtk.vtkPolyDataMapper()
    # mapper_text_origin.SetInputConnection(text_origin.GetOutputPort())
    # # actor_text_origin = vtk.vtkActor()
    # actor_text_origin = vtk.vtkFollower()
    # actor_text_origin.SetCamera(_renderer.GetActiveCamera())
    # actor_text_origin.SetMapper(mapper_text_origin)
    # actor_text_origin.SetScale(0.2, 0.2, 0.2)
    # actor_text_origin.AddPosition(0, -0.1, 0)


    # _renderer.AddActor(actor_text_origin)
    # _renderer.ResetCamera()

    # iren.Initialize()  # will be called by Start() autometically
    if has_picker:
        iren.AddObserver('MouseMoveEvent', MoveCursor)
    iren.Start()


def get_actor(vtk_source, color=color_diffuse, opacity=1.0,
              has_scalar_visibility=False, has_decimator=False):
    """
    Set `scalar_visibility` be `True` makes `color` unavailable.
    :return: a vtkActor
    """
    if has_decimator:
        # Reduce the number of triangles
        decimator = vtk.vtkDecimatePro()
        decimator.SetInputConnection(vtk_source.GetOutputPort())
        # decimator.SetInputData(vtk_source)
        decimator.SetFeatureAngle(60)
        decimator.MaximumIterations = 1
        decimator.PreserveTopologyOn()
        decimator.SetMaximumError(0.0002)
        decimator.SetTargetReduction(1)
        decimator.SetErrorIsAbsolute(1)
        decimator.SetAbsoluteError(0.0002)
        decimator.ReleaseDataFlagOn()

    # Generate Normals
    normals = vtk.vtkPolyDataNormals()
    if has_decimator:
        normals.SetInputConnection(decimator.GetOutputPort())
    else:
        normals.SetInputConnection(vtk_source.GetOutputPort())
    normals.SetFeatureAngle(60.0)
    normals.ReleaseDataFlagOn()

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(normals.GetOutputPort())
    stripper.ReleaseDataFlagOn()

    mapper = vtk.vtkPolyDataMapper()
    # mapper.SetInputConnection(vtk_source.GetOutputPort())
    mapper.SetInputConnection(stripper.GetOutputPort())
    mapper.SetScalarVisibility(has_scalar_visibility)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetDiffuseColor(color)
    actor.GetProperty().SetSpecular(0.3)
    actor.GetProperty().SetSpecularPower(20)
    actor.GetProperty().SetInterpolation(2)
    # actor.GetProperty().SetRepresentation(2)
    # actor.GetProperty().SetEdgeVisibility(True)
    # actor.GetProperty().SetOpacity(opacity)

    if opacity < 1.0:
        if is_depth_peeling_supported(render_window, renderer, True):
            setup_evn_for_depth_peeling(render_window, renderer, max_peels, occlusion)
            actor.GetProperty().SetOpacity(opacity)
        else:
            print "Depth Peeling is not supported."

    return actor


def implicit_fitting(data):
    """
    Find the fitting according to input dataset
    :param data: point_num*3 matrix, every row is a 3D points
    :return: colume vector: point_num*1
    """

    num_points = len(data)

    A = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(i+1, num_points):
            A[i, j] = np.linalg.norm(data[i, :] - data[j, :])
            A[j, i] = A[i, j]

    dx = data[:, 0]
    dy = data[:, 1]
    dz = data[:, 2]

    dx = dx.reshape(num_points, 1)
    dy = dy.reshape(num_points, 1)
    dz = dz.reshape(num_points, 1)

    B = np.hstack((np.ones((num_points, 1)),
                   2 * dx, 2 * dy, 2 * dz,
                   2 * dx * dy, 2 * dx * dz, 2 * dy * dz,
                   dx * dx, dy * dy, dz * dz)
                  )
    # B_t = np.transpose(B)
    # M = [[A, B], [B_t, np.zeros((10, 10))]]  # It is symmetric

    M_t1 = np.concatenate((A, B.T))
    M_t2 = np.concatenate((B, np.zeros((10, 10))))
    M = np.concatenate((M_t1, M_t2), axis=1)  # It is sysmmetric

    k = 1000
    C0 = np.zeros((3, 3))
    C1 = np.diag([-k] * 3)
    C2 = np.ones((3, 3)) * (k - 2) / 2.0
    np.fill_diagonal(C2, -1)
    C11 = np.concatenate((C1, C0))
    C22 = np.concatenate((C0, C2))
    C = np.concatenate((C11, C22), axis=1)

    M11 = M[0:-6, 0:-6]  # (num_points) * (num_points)
    M12 = M[0:-6, -6:]  # (num_point-6) * 6
    M22 = M[-6:, -6:]  # 6 * 6 zero matrix

    pinvM11 = np.linalg.pinv(M11)
    M0 = np.dot(pinvM11, M12)
    M00 = M22 - np.dot(M12.T, M0)

    if np.all(np.linalg.eigvals(M00)) > 0:  # Positive Definite
        eigen_value, eigen_vec = np.linalg.eig(M00)
    else:
        M00 = np.dot(M00.T * M00)

    eigen_value, eigen_vec = sci.linalg.eig(M00, C)
    # D = np.diag(eigen_value)
    max_eigen_value = np.amax(eigen_value)
    max_eigen_idx = np.argmax(eigen_value)

    # Find the fitting
    V1 = eigen_vec[:, max_eigen_idx]
    V0 = np.dot(-M0, V1)
    V = np.hstack((V0, V1))
    V = V.reshape(num_points+10, 1)

    return V


def ndarray2vtkImageData(ndarray, cast_type=11, spacing=[1, 1, 1]):
    """
    Convert a NumPy array to a vtkImageData, with a default casting type VTK_DOUBLE
    :param ndarray: input NumPy array, can be 3D array
    :param cast_type: 11 means VTK_DOUBLE
    :return: a vtkImageData
    """
    # Convert numpy array to VTK array (vtkDoubleArray)
    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=ndarray.transpose(2, 1, 0).ravel(),
        deep=True,
        array_type=vtk.VTK_DOUBLE)

    # Convert the VTK array to vtkImageData
    img_vtk = vtk.vtkImageData()
    img_vtk.SetDimensions(ndarray.shape)
    img_vtk.SetSpacing(spacing[::-1])  # Note the order should be reversed!
    img_vtk.GetPointData().SetScalars(vtk_data_array)  # is a vtkImageData

    # casting
    cast = vtk.vtkImageCast()
    cast.SetInputData(img_vtk)
    cast.SetOutputScalarType(cast_type)
    cast.Update()

    return cast.GetOutput()  # vtkImageData


def add_point(_renderer, position=[0, 0, 0], color=[0.4, 0.4, 0.4], radius=0.2):
    _point = vtk.vtkSphereSource()
    _point.SetCenter(position)
    _point.SetRadius(radius)
    _point.SetPhiResolution(10)
    _point.SetThetaResolution(10)

    _mapper_point = vtk.vtkPolyDataMapper()
    _mapper_point.SetInputConnection(_point.GetOutputPort())

    _actor_point = vtk.vtkActor()
    _actor_point.SetMapper(_mapper_point)
    _actor_point.GetProperty().SetColor(color)

    _renderer.AddActor(_actor_point)


def add_text(_renderer, position, text="TEXT", color=[0.5, 0.5, 0.5], scale=0.1):
    # Create text with the x-y-z coordinate system
    _text = vtk.vtkVectorText()
    _text.SetText(text)
    mapper_text = vtk.vtkPolyDataMapper()
    mapper_text.SetInputConnection(_text.GetOutputPort())
    # actor_text_origin = vtk.vtkActor()
    actor_text = vtk.vtkFollower()
    # actor_text.SetCamera(_renderer.GetActiveCamera())
    actor_text.SetMapper(mapper_text)
    actor_text.SetScale(scale, scale, scale)
    actor_text.GetProperty().SetColor(color)
    actor_text.AddPosition([sum(x) for x in zip(position, [0, -0.1, 0])])  # plus of 2 arrays

    _renderer.AddActor(actor_text)


def setup_evn_for_depth_peeling(_render_window, _renderer, max_no_peels, occlusion_ratio):
    if not(_render_window) or not(_renderer):
        return False

    _render_window.SetAlphaBitPlanes(True)
    _render_window.SetMultiSamples(0)
    _renderer.SetUseDepthPeeling(True)
    _renderer.SetMaximumNumberOfPeels(max_no_peels)
    _renderer.SetOcclusionRatio(occlusion_ratio)

    return True


def is_depth_peeling_supported(_render_window, _renderer, _do_it_off_screen):
    if not(_render_window) or not(_renderer):
        return False

    success = True

    original_offscreen_rendering = bool(_render_window.GetOffScreenRendering())
    original_alpha_bit_planes = bool(_render_window.GetAlphaBitPlanes())
    original_multi_sample = _render_window.GetMultiSamples()
    original_use_depth_peeling = bool(_renderer.GetUseDepthPeeling())
    original_max_peels = _renderer.GetMaximumNumberOfPeels()
    original_occlusion_ratio = _renderer.GetOcclusionRatio()

    # Active off screen rendering on demand
    _render_window.SetOffScreenRendering(_do_it_off_screen)

    # Setup environment for depth peeling
    success = success and setup_evn_for_depth_peeling(
        _render_window, _renderer, 100, 0.1)

    # Recover original state
    _render_window.SetOffScreenRendering(original_offscreen_rendering)
    _render_window.SetAlphaBitPlanes(original_alpha_bit_planes)
    _render_window.SetMultiSamples(original_multi_sample)
    _renderer.SetUseDepthPeeling(original_use_depth_peeling)
    _renderer.SetMaximumNumberOfPeels(original_max_peels)
    _renderer.SetOcclusionRatio(original_occlusion_ratio)

    return success



matlab_file = "/Users/Quentan/Box Sync/VTK_Python/Algorithms/data/patella.mat"
# matlab_file = "/Users/Quentan/Box Sync/VTK_Python/Algorithms/data/head.mat"
# matlab_file = "/Users/Quentan/Box Sync/VTK_Python/Algorithms/data/tibia.mat"

data = loadmat(matlab_file)
data = data['data']
v = implicit_fitting(data)
w = v[-10:]

num_points = len(data)

# dx = data[:, 0]
# dy = data[:, 1]
# dz = data[:, 2]

# dx = dx.reshape(num_points, 1)
# dy = dy.reshape(num_points, 1)
# dz = dz.reshape(num_points, 1)

# Get the fitting surface using meshgrid
# min_x = np.amin(dx)
# max_x = np.amax(dx)
# min_y = np.amin(dy)
# max_y = np.amax(dy)
# min_z = np.amin(dz)
# max_z = np.amax(dz)

# step = 50
# offset = 0.1
# step_x = np.linspace(min_x - offset, max_x + offset, step)
# step_y = np.linspace(min_y - offset, max_y + offset, step)
# step_z = np.linspace(min_z - offset, max_z + offset, step)
# spacing = [max_x-min_x, max_y-min_y, max_z-min_z]

# Another way of stepping
data_min = data.min(0)
data_max = data.max(0)

step = 0.05
offset = 0.1
step_x = np.arange(data_min[0]-offset, data_max[0]+offset, step)
step_y = np.arange(data_min[1]-offset, data_max[1]+offset, step)
step_z = np.arange(data_min[2]-offset, data_max[2]+offset, step)

spacing = data_max - data_min


[x, y, z] = np.meshgrid(step_x, step_y, step_z)

poly = w[0]*np.ones((x.shape)) + \
       2*w[1]*x + 2*w[2]*y + 2*w[3]*z + \
       2*w[4]*x*y + 2*w[5]*x*z + 2*w[6]*y*z + \
       w[7]*x*x + w[8]*y*y + w[9]*z*z

radial = np.zeros((x.shape))
for i in range(num_points):
    # radial = radial + v[i]*np.sqrt((x-dx[i])**2 + (y-dy[i])**2 + (z-dz[i])**2)
    radial = radial + v[i]*np.sqrt(
        (x-data[i, 0])**2 + (y-data[i, 1])**2 + (z-data[i, 2])**2)

obj = poly + radial

# Draw points
for i in range(num_points):
    add_point(renderer, data[i, :], color=[1, 0, 0], radius=0.01)
    # add_text(renderer, data[i, :], text=str(i+1))

reader = ndarray2vtkImageData(obj)
# reader = ndarray2vtkImageData(poly)

dims = reader.GetDimensions()
bounds = reader.GetBounds()
spacing = reader.GetSpacing()
origin = reader.GetOrigin()

# Convert the vtkImageData to vtkImplicitFunction
implicit_volume = vtk.vtkImplicitVolume()
implicit_volume.SetVolume(reader)

sample = vtk.vtkSampleFunction()
sample.SetImplicitFunction(implicit_volume)
sample.SetModelBounds(bounds)
sample.SetSampleDimensions(dims)
# sample.SetSampleDimensions(sample_dims)
sample.ComputeNormalsOff()

contour = vtk.vtkContourFilter()
contour.SetInputConnection(sample.GetOutputPort())
# contour.SetValue(0, contour_bone)
contour.SetValue(0, contour_manual)

actor = get_actor(contour)
# renderer.AddActor(actor)
renderer.SetBackground(0.2, 0.3, 0.4)
vtk_show(renderer)
