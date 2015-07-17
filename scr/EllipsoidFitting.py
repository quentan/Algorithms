#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Li's Ellipsoid Fitting Algorithm
This test remains a state of "smoothness issues", see the note on 'Daily Plan (Fri 10/07/2015)'
"""

import numpy as np
import vtk
from vtk.util import numpy_support

contour_manual = 1.0
color_diffuse = [1.0, 1.0, 0.0]
sample_dims = [100] * 3

# Smoothness for visualisation
smooth_iteration = 100
smooth_factor = 0.1
smooth_angle = 60

# Initialisation
render_window = vtk.vtkRenderWindow()
iren = vtk.vtkRenderWindowInteractor()
renderer = vtk.vtkRenderer()

def vtk_show(_renderer, window_name='VTK Show Window', width=640, height=480, has_picker=False):
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


def get_actor(vtk_source, color=color_diffuse, opacity=1.0, has_scalar_visibility=False):
    """
    Set `scalar_visibility` be `True` makes `color` unavailable.
    :return: a vtkActor
    """
    # Reduce the number of triangles
    decimator = vtk.vtkDecimatePro()
    decimator.SetInputConnection(vtk_source.GetOutputPort())
    # decimator.SetInputData(vtk_source)
    decimator.SetFeatureAngle(60)
    decimator.MaximumIterations = 1
    decimator.PreserveTopologyOn()
    decimator.SetMaximumError(0.0002)
    decimator.SetTargetReduction(0.3)
    decimator.SetErrorIsAbsolute(1)
    decimator.SetAbsoluteError(0.0002)
    decimator.ReleaseDataFlagOn()

    # Smooth the triangle vertices
    smoother = vtk.vtkSmoothPolyDataFilter()
    # smoother.SetInputConnection(decimator.GetOutputPort())
    smoother.SetInputConnection(vtk_source.GetOutputPort())
    smoother.SetNumberOfIterations(smooth_iteration)
    smoother.SetRelaxationFactor(smooth_factor)
    smoother.SetFeatureAngle(smooth_angle)
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOff()
    smoother.SetConvergence(0)
    smoother.ReleaseDataFlagOn()

    # Generate Normals
    normals = vtk.vtkPolyDataNormals()
    # normals.SetInputConnection(vtk_source.GetOutputPort())
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.SetFeatureAngle(60.0)
    normals.ReleaseDataFlagOn()

    stripper = vtk.vtkStripper()
    stripper.SetInputConnection(normals.GetOutputPort())
    stripper.ReleaseDataFlagOn()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(vtk_source.GetOutputPort())
    # mapper.SetInputConnection(stripper.GetOutputPort())
    mapper.SetScalarVisibility(has_scalar_visibility)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetDiffuseColor(color)
    actor.GetProperty().SetSpecular(0.3)
    actor.GetProperty().SetSpecularPower(20)
    # actor.GetProperty().SetOpacity(opacity)

    if opacity < 1.0:
        if is_depth_peeling_supported(render_window, renderer, True):
            setup_evn_for_depth_peeling(render_window, renderer, max_peels, occlusion)
            actor.GetProperty().SetOpacity(opacity)
        else:
            print "Depth Peeling is not supported."

    return actor


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


def find_fit(S):
    """
    Find the fitting accroding to coefficient matrix
    :param S: Coefficent matrix: S=D*D'. D is 10*numPoints matrix
    :return: colume vector: 10*1
    """

    # Consgtraint Matrix C:
    C1 = np.diag([-1, -1, -1, -4, -4, -4])
    C2 = np.ones((6, 6))
    np.fill_diagonal(C2, 0)  # Change C2's diagonal values to all 0
    C = C1 + C2

    # Solve generalised eigensystem
    S11 = S[0:6, 0:6]
    S12 = S[0:6, 6:10]
    S22 = S[6:10, 6:10]

    # A = S11 - S12 * np.linalg.pinv(S22) * np.transpose(S12)  # Syntax error for arrays
    A = S11 - np.dot(S12, np.dot(np.linalg.pinv(S22), np.transpose(S12)))

    # CA = np.linalg.inv(C) * A
    CA = np.dot(np.linalg.inv(C), A)
    geval, gevec = np.linalg.eig(CA)

    # Find the largest eigenvalue, which is the only positve one
    maxVal = np.amax(geval)
    maxIdx = np.argmax(geval)

    # Find the fitting
    v1 = gevec[:, maxIdx]
    v2 = - np.dot(np.dot(np.linalg.pinv(S22), np.transpose(S12)), v1)
    v = np.hstack((v1, v2))
    v = v.reshape(10, 1)

    return v

num_points = 20
data = np.random.standard_normal((num_points, 3))

dx = data[:, 0]
dy = data[:, 1]
dz = data[:, 2]

dx = dx.reshape(num_points, 1)
dy = dy.reshape(num_points, 1)
dz = dz.reshape(num_points, 1)

D = np.hstack((dx * dx, dy * dy, dz * dz,
              2 * dy * dz, 2 * dx * dz, 2 * dx * dy,
              2 * dx, 2 * dy, 2 * dz,
              np.ones((num_points, 1)))
              )
D_t = np.transpose(D)
S = np.dot(D_t, D)

v = find_fit(S)

coefficient = [1, 1, 1, 2, 2, 2, 2, 2, 2, 1]
v2 = [x*y for x, y in zip(v, coefficient)]

# Draw the points
for i in range(num_points):
    add_point(renderer, data[i, :], color=[1, 0, 0], radius=0.05)
    add_text(renderer, data[i, :], text=str(i+1))


# Get the fitting surface using meshgrid
min_x = np.amin(dx)
max_x = np.amax(dx)
min_y = np.amin(dy)
max_y = np.amax(dy)
min_z = np.amin(dz)
max_z = np.amax(dz)

step = 102
offset = 0.1
step_x = np.linspace(min_x - offset, max_x + offset, step)
step_y = np.linspace(min_y - offset, max_y + offset, step)
step_z = np.linspace(min_z - offset, max_z + offset, step)

[x, y, z] = np.meshgrid(step_x, step_y, step_z)

solid_obj = v[0]*x*x + v[1]*y*y + v[2]*z*z + \
            2*v[3]*y*z + 2*v[4]*x*z + 2*v[5]*x*y + \
            2*v[6]*x + 2*v[7]*y + 2*v[8]*z + \
            v[9]*np.ones((x.shape))

# solid_obj = v[0]*x*x + v[1]*y*y + v[2]*z*z + \
#             v[3]*x*y + v[4]*y*z + v[5]*x*z + \
#             v[6]*x + v[7]*y + v[8]*z + \
#             v[9]*np.ones((x.shape))

# Convert numpy array to VTK array (vtkArray)
# vtk_data_array = numpy_support.numpy_to_vtk(
#     num_array=solid_obj.transpose(2, 1, 0).ravel(),
#     deep=True,
#     array_type=vtk.VTK_DATA_OBJECT)

vtk_data_array = numpy_support.numpy_to_vtk(
    num_array=solid_obj.transpose(2, 1, 0).ravel(),
    deep=True,
    array_type=vtk.VTK_DOUBLE)

# spacing = [1, 1, 1]  # This is default value
spacing = [max_x-min_x, max_y-min_y, max_z-min_z]

# Convert vtkArray to vtkImageData
img_vtk = vtk.vtkImageData()
img_vtk.SetDimensions(solid_obj.shape)
# test = img_vtk.GetSpacing()
# img_vtk.SetSpacing(list(reversed(spacing)))
img_vtk.SetSpacing(spacing[::-1])  # Note the order should be reversed!
img_vtk.GetPointData().SetScalars(vtk_data_array)  # is a vtkImageData

# casting
cast_type = 10  # 4 is a `short int`, will lead to low precision result
cast = vtk.vtkImageCast()
cast.SetInputData(img_vtk)
# cast.SetInputConnection(reader.GetOutputPort())
cast.SetOutputScalarType(cast_type)
cast.Update()

reader = cast.GetOutput()  # vtkImageData

dims = [0] * 3
bounds = [0.0] * 6
spacing = [0] * 3
origin = [0.0] * 3
dims = reader.GetDimensions()
bounds = reader.GetBounds()
spacing = reader.GetSpacing()
origin = reader.GetOrigin()

scalar_min = reader.GetScalarTypeMin()
scalar_max = reader.GetScalarTypeMax()

# TEST: Generate a quadric directly from fitting result
coe = [.5, 1, .2, 0, .1, 0, 0, .2, 0, 0]
quadric = vtk.vtkQuadric()
quadric.SetCoefficients(v2)

# Convert the vtkImageData to vtkImplicitFunction
implicit_volume = vtk.vtkImplicitVolume()
implicit_volume.SetVolume(reader)
# implicit_volume.SetVolume(vtk_data_array)
# implicit_volume.SetVolume(img_vtk)

sample = vtk.vtkSampleFunction()
sample.SetImplicitFunction(implicit_volume)
# sample.SetImplicitFunction(quadric)
# sample.SetModelBounds(-2, 2, -2, 2, -2, 2)
# sample.SetModelBounds(min_x, max_x, min_y, max_y, min_z, max_z)
sample.SetModelBounds(bounds)
# dims = [sum(x) for x in zip(dims, [1] * 3)]
sample.SetSampleDimensions(dims)
# sample.SetSampleDimensions(sample_dims)
sample.ComputeNormalsOff()

contour = vtk.vtkContourFilter()
contour.SetInputConnection(sample.GetOutputPort())
# contour.SetValue(0, contour_bone)
contour.SetValue(0, contour_manual)

actor = get_actor(contour)
renderer.AddActor(actor)
# renderer.SetBackground(1, 1, 1)
vtk_show(renderer)
