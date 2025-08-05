# Exploring Side_Panel sessions from the 2025 bootcamp
#
# Note that the progressObj and zoneTuples will need to be drawn
# from the guidelet currently, and the guidelet needs to have the
# correct head model selected on launch.  This is not optimal,
# and serialized versions of these would be better, but are not
# initially available.

import numpy as np


def getModelDistanceFunction(modelNode, world=True):
    """Get a function which can return the signed distance to the closest
    point on the supplied model node. This returns a function so that it
    can be used for many different points without having to be re-created
    every time. If the "world" flag is True, then any soft transforms are
    applied to the model before distance calculation; if False, the
    model points are treated as though they are untransformed.
    """
    # Handle if model node has a parent soft transform
    if modelNode.GetParentTransformNode() and world:
        transformModelToWorld = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            modelNode.GetParentTransformNode(), None, transformModelToWorld
        )
        polyTransformToWorld = vtk.vtkTransformPolyDataFilter()
        polyTransformToWorld.SetTransform(transformModelToWorld)
        polyTransformToWorld.SetInputData(modelNode.GetPolyData())
        polyTransformToWorld.Update()
        polyData = polyTransformToWorld.GetOutput()
    else:
        # Not transformed or ignoring transform, can just use as is
        polyData = modelNode.GetPolyData()

    if not polyData or polyData.GetNumberOfPoints() == 0:
        modelName = modelNode.GetName() if modelNode.GetName() else "Unnamed"
        print(
            f"Error: Model node '{modelName}' does not contain valid polydata or has no points."
        )
        return None
    distanceFilter = vtk.vtkImplicitPolyDataDistance()
    distanceFilter.SetInput(polyData)
    # Check whether model normals are inverted by checking 3 points, at least two of which
    # should be outside the model
    pt0 = [0, 0, 0]
    pt1 = [100, 100, 100]
    pt2 = [-100, -100, -100]
    signs = [np.sign(distanceFilter.EvaluateFunction(pt)) for pt in [pt0, pt1, pt2]]
    if np.sum(signs) < 0:
        # Model normals are inverted, negate distances
        negateFlag = True
    else:
        negateFlag = False

    def distanceFcn(point):
        if not isinstance(point, (list, tuple, np.ndarray)) or len(point) != 3:
            print(
                "Error: Point must be a list, tuple, or numpy array of 3 coordinates (x, y, z)."
            )
            return None
        try:
            # Ensure point coordinates are floats for VTK
            point_coords = [float(p) for p in point]
        except ValueError:
            print("Error: Point coordinates must be numeric.")
            return None
        # Evaluate
        signedDistance = distanceFilter.EvaluateFunction(point)
        if negateFlag:
            signedDistance = -1 * signedDistance
        return signedDistance

    # Return the created distance function for this model
    return distanceFcn


def getClosestPointFunction(modelNode, world=True):
    """Get a function which can return the closest point and signed distance
    to the closest point on the supplied model node. This returns a function so that it
    can be used for many different points without having to be re-created
    every time. If the "world" flag is True, then any soft transforms are
    applied to the model before distance calculation; if False, the
    model points are treated as though they are untransformed.
    """
    # Handle if model node has a parent soft transform
    if modelNode.GetParentTransformNode() and world:
        transformModelToWorld = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            modelNode.GetParentTransformNode(), None, transformModelToWorld
        )
        polyTransformToWorld = vtk.vtkTransformPolyDataFilter()
        polyTransformToWorld.SetTransform(transformModelToWorld)
        polyTransformToWorld.SetInputData(modelNode.GetPolyData())
        polyTransformToWorld.Update()
        polyData = polyTransformToWorld.GetOutput()
    else:
        # Not transformed or ignoring transform, can just use as is
        polyData = modelNode.GetPolyData()

    if not polyData or polyData.GetNumberOfPoints() == 0:
        modelName = modelNode.GetName() if modelNode.GetName() else "Unnamed"
        print(
            f"Error: Model node '{modelName}' does not contain valid polydata or has no points."
        )
        return None
    distanceFilter = vtk.vtkImplicitPolyDataDistance()
    distanceFilter.SetInput(polyData)
    # Check whether model normals are inverted by checking 3 points, at least two of which
    # should be outside the model
    pt0 = [0, 0, 0]
    pt1 = [100, 100, 100]
    pt2 = [-100, -100, -100]
    signs = [np.sign(distanceFilter.EvaluateFunction(pt)) for pt in [pt0, pt1, pt2]]
    if np.sum(signs) < 0:
        # Model normals are inverted, negate distances
        negateFlag = True
    else:
        negateFlag = False

    def closestPtFcn(point):
        if not isinstance(point, (list, tuple, np.ndarray)) or len(point) != 3:
            print(
                "Error: Point must be a list, tuple, or numpy array of 3 coordinates (x, y, z)."
            )
            return None
        try:
            # Ensure point coordinates are floats for VTK
            point_coords = [float(p) for p in point]
        except ValueError:
            print("Error: Point coordinates must be numeric.")
            return None
        # Evaluate
        closest = np.zeros(3)
        signedDistance = distanceFilter.EvaluateFunctionAndGetClosestPoint(
            point, closest
        )
        if negateFlag:
            signedDistance = -1 * signedDistance
        return closest, signedDistance

    # Return the created distance function for this model
    return closestPtFcn


def showLink(pt, closestPtFcn, linkMarkupNode=None):
    if linkMarkupNode is None:
        linkMarkupNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsLineNode", "Link"
        )
        # set up display properties
        dn = linkMarkupNode.GetDisplayNode()
        dn.SetGlyphScale(1)
        dn.SetUseGlyphScale(True)
    # Get closest point
    closest = closestPtFcn(pt)[0]
    #
    # linkMarkupNode.RemoveAllControlPoints()
    controlPointsArr = np.vstack((pt, closest))
    slicer.util.updateMarkupsControlPointsFromArray(linkMarkupNode, controlPointsArr)
    return linkMarkupNode


sessionsDir = r"C:\Users\mike.bindschadler@seattlechildrens.org\OneDrive - SCH\Airway4D\Temp\TrackerFiles\2025\Sessions_Backup\Side_Panel_Tuesday"

progObj = slicer.modules.ExampleGuideletWidget.guideletInstance.progressObj
zoneTuples = slicer.modules.ExampleGuideletWidget.guideletInstance.zoneTuples

# from HelperClasses import ScopeRun, loadAllScopeRunsFromDirectory

listOfScopeRuns = loadAllScopeRunsFromDirectory(sessionsDir)

for sr in listOfScopeRuns:
    sr.analyze(progObj, zoneTuples)
    if sr.valid:
        print(f"Score: {sr.score}")
    else:
        print(f"Invalid run")

validRuns = [sr for sr in listOfScopeRuns if sr.valid]


# Assemble all points into a list  Color points by inside/outside airway lumen
allPos = np.empty((0, 3))
for sr in validRuns:
    allPos = np.vstack([allPos, sr.positions])

# Create vtk arrays needed for model node
points = vtk.vtkPoints()
vertices = vtk.vtkCellArray()
colors = vtk.vtkFloatArray()
colors.SetName("ColorIdx")

lumenModel = getNode("Side_AirwayLumen")

zoneDict = {zt[0]: {"modelNode": zt[1], "triggerDist": zt[2]} for zt in zoneTuples}
zoneDict["Lumen"] = {"modelNode": lumenModel, "triggerDist": 0}

vtkArrList = []
modelDistFcnList = []
for zoneName, modelDistDict in zoneDict.items():
    modelNode = modelDistDict["modelNode"]
    distFcn = getModelDistanceFunction(modelNode)
    modelDistDict["distanceFcn"] = distFcn
    vtkArr = vtk.vtkFloatArray()
    vtkArr.SetName(f"{zoneName}Distance")
    vtkArrList.append(vtkArr)
    modelDistFcnList.append(distFcn)

for pos in allPos:
    pointID = points.InsertNextPoint(pos)
    cellID = vertices.InsertNextCell(1)
    vertices.InsertCellPoint(pointID)
    # Determine whether in or out and color appropriately
    colorIdx = 1
    colors.InsertNextValue(colorIdx)
    for vtkArr, distFcn in zip(vtkArrList, modelDistFcnList):
        vtkArr.InsertNextValue(distFcn(pos))


pointsPolyData = vtk.vtkPolyData()
pointsPolyData.SetPoints(points)
pointsPolyData.SetVerts(vertices)
pointsPolyData.GetPointData().AddArray(colors)
for vtkArr in vtkArrList:
    pointsPolyData.GetPointData().AddArray(vtkArr)


sphereSource = vtk.vtkSphereSource()

glyphFilter = vtk.vtkGlyph3D()
glyphFilter.SetSourceConnection(sphereSource.GetOutputPort())
glyphFilter.SetInputData(pointsPolyData)

modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "AllPoints")
modelNode.CreateDefaultDisplayNodes()
modelDisplay = modelNode.GetDisplayNode()
# Connect to glyph output
modelNode.SetPolyDataConnection(glyphFilter.GetOutputPort())

modelDisplay.SetAndObserveColorNodeID("vtkMRMLColorTableNodeFileColdToHotRainbow.txt")
# Color table names can be found at https://apidocs.slicer.org/master/classvtkMRMLColorLogic.html
modelDisplay.SetScalarVisibility(True)
modelDisplay.SetActiveScalarName("LumenDistance")
modelDisplay.SetScalarRange(0, 3)
modelDisplay.SetThresholdEnabled(True)
modelDisplay.SetThresholdRange(0, 5)

## Explore where bad distances are coming from in coughZone
