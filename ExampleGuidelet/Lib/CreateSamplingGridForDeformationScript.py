def createSamplingGridForDeformation(node, spacingMm=10.0):
    """
    Constructs a regular 3D grid of sampling points around a node. The
    node must have a GetBounds() method.

    Args:
        modelNode (vtkMRMLModelNode): The model node around which to create the grid.
        margin (float, optional): The margin to add around the model's bounding box in mm. Defaults to 5.0.
        spacing (float, optional): The desired spacing between grid points in mm. Defaults to 10.0.

    Returns:
        numpy array of grid points.
    """
    if not node:
        raise Exception("No node provided!")
    if not hasattr(node, "GetBounds"):
        raise Exception("Input does not have required GetBounds() method")

    # Get the bounding box of the model in RAS coordinates
    bounds = [0.0] * 6  # xmin, xmax, ymin, ymax, zmin, zmax
    node.GetBounds(bounds)

    xmin = bounds[0]
    xmax = bounds[1]
    ymin = bounds[2]
    ymax = bounds[3]
    zmin = bounds[4]
    zmax = bounds[5]

    x_coords = [x for x in np.arange(xmin, xmax, spacingMm)]
    y_coords = [y for y in np.arange(ymin, ymax, spacingMm)]
    z_coords = [z for z in np.arange(zmin, zmax, spacingMm)]

    # Generate grid points
    gridPointsNumpy = np.array(
        tuple([x, y, z] for x in x_coords for y in y_coords for z in z_coords)
    )

    # return gridPolyData
    return gridPointsNumpy


# imNode = getNode("Chest_1mm")
modelNode = getNode("AirwayLumenModel")
gridPointsNumpy = createSamplingGridForDeformation(modelNode, 20)

markupsNode = slicer.mrmlScene.AddNewNodeByClass(
    "vtkMRMLMarkupsFiducialNode", "ForDeformation"
)
slicer.util.updateMarkupsControlPointsFromArray(markupsNode, gridPointsNumpy)
