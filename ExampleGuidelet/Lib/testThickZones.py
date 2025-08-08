import numpy as np

sessionsDir = r"C:\Users\mike.bindschadler@seattlechildrens.org\OneDrive - SCH\Airway4D\Temp\TrackerFiles\2025\Sessions_Backup\Chest_Panel_Tuesday"

# progObj = slicer.modules.ExampleGuideletWidget.guideletInstance.progressObj
# zoneTuples = slicer.modules.ExampleGuideletWidget.guideletInstance.zoneTuples

# from HelperClasses import ScopeRun, loadAllScopeRunsFromDirectory

listOfScopeRuns = loadAllScopeRunsFromDirectory(sessionsDir)
allPos = np.empty((0, 3))
for sr in listOfScopeRuns:
    # includes invalid
    allPos = np.vstack([allPos, sr.positions])

# Create vtk arrays needed for model node
points = vtk.vtkPoints()
vertices = vtk.vtkCellArray()
colors = vtk.vtkFloatArray()
colors.SetName("ColorIdx")

zoneDict = {
    "CoughZone": {"modelNode": getNode("CoughZone_Thick"), "triggerDist": 0},
    "GagZone": {"modelNode": getNode("GagZone_Thick"), "triggerDist": 0},
}

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
modelDisplay.SetActiveScalarName("CoughZoneDistance")
modelDisplay.SetScalarRange(0, 3)
modelDisplay.SetThresholdEnabled(True)
modelDisplay.SetThresholdRange(0, 5)
