import os
from __main__ import vtk, qt, ctk, slicer

from SlicerGuideletBase import GuideletLoadable, GuideletLogic, GuideletTest, GuideletWidget
from SlicerGuideletBase import Guidelet
import logging
import time
from Lib.HelperClasses import Session, Recording, ScopeRun


class ExampleGuidelet(GuideletLoadable):
  """Uses GuideletLoadable class, available at:
  """

  def __init__(self, parent):
    GuideletLoadable.__init__(self, parent)
    self.parent.title = "ExampleGuidelet"
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["YOUR NAME"]
    self.parent.helpText = """ SOME HELP AND A LINK TO YOUR WEBSITE """
    self.parent.acknowledgementText = """ THANKS TO ... """


class ExampleGuideletWidget(GuideletWidget):
  """Uses GuideletWidget base class, available at:
  """

  def __init__(self, parent = None):
    GuideletWidget.__init__(self, parent)


  def setup(self):
    GuideletWidget.setup(self)
    fileDir = os.path.dirname(__file__)
    iconPathRecord = os.path.join(fileDir, 'Resources', 'Icons', 'icon_Record.png')
    iconPathStop = os.path.join(fileDir, 'Resources', 'Icons', 'icon_Stop.png')

    if os.path.isfile(iconPathRecord):
      self.recordIcon = qt.QIcon(iconPathRecord)
    if os.path.isfile(iconPathStop):
      self.stopIcon = qt.QIcon(iconPathStop)


  def addLauncherWidgets(self):
    GuideletWidget.addLauncherWidgets(self)


  def onConfigurationChanged(self, selectedConfigurationName):
    GuideletWidget.onConfigurationChanged(self, selectedConfigurationName)
    #settings = slicer.app.userSettings()


  def addBreachWarningLightPreferences(self):
    pass


  def onBreachWarningLightChanged(self, state):
    pass


  def createGuideletInstance(self):
    return ExampleGuideletGuidelet(None, self.guideletLogic, self.selectedConfigurationName)


  def createGuideletLogic(self):
    return ExampleGuideletLogic()
  
  def onStartStopRecordingClicked(self):
    """ originally Copied from UltraSound.py"""
    if self.startStopRecordingButton.isChecked():
      self.startStopRecordingButton.setText("  Stop Recording")
      self.startStopRecordingButton.setIcon(self.stopIcon)
      self.startStopRecordingButton.setToolTip("Recording is being started...")
      if self.captureDeviceName  != '':
        # Important to save as .mhd because that does not require lengthy finalization (merging into a single file)
        recordPrefix = self.parameterNode.GetParameter('RecordingFilenamePrefix')
        recordExt = self.parameterNode.GetParameter('RecordingFilenameExtension')
        self.recordingFileName =  recordPrefix + time.strftime("%Y%m%d-%H%M%S") + recordExt

        logging.info("Starting recording to: {0}".format(self.recordingFileName))

        self.plusRemoteNode.SetCurrentCaptureID(self.captureDeviceName)
        self.plusRemoteNode.SetRecordingFilename(self.recordingFileName)
        self.plusRemoteLogic.StartRecording(self.plusRemoteNode)

    else:
      self.startStopRecordingButton.setText("  Start Recording")
      self.startStopRecordingButton.setIcon(self.recordIcon)
      self.startStopRecordingButton.setToolTip( "Recording is being stopped..." )
      if self.captureDeviceName  != '':
        logging.info("Stopping recording")
        self.plusRemoteNode.SetCurrentCaptureID(self.captureDeviceName)
        self.plusRemoteLogic.StopRecording(self.plusRemoteNode)

HEAD_SENSOR_TRANSFORM_POSITION_IN_HIERARCHY = 1
SCOPE_SENSOR_TRANSFORM_POSITION_IN_HIERARCHY = 2
DEFAULT_LEAF_TRANSFORM_NODE_NAME = 'Extra'
moduleDir = os.path.dirname(__file__)
DEFAULT_AIRWAYZONE_SEGMENTATION = os.path.join(moduleDir, "Resources","airwayZoneSegmentation.seg.nrrd")

class ExampleGuideletLogic(GuideletLogic):
  """Uses GuideletLogic base class, available at:
  """ #TODO add path


  def __init__(self, parent = None):
    GuideletLogic.__init__(self, parent)

  def createSessionFile(self, headerList, currentSessionFilePath):
    raise(Exception('ExampleGuidelet.createSessionFile() accessed, but functionality moved to Session objects!'))
    """Create a file to hold the results of a session for a single user.  This should be
    called whenever new user information is saved.  Whenever recordings are started/finished,
    a line should be added to the session file saying where the file is saved and what the 
    file name is."""
    lines = [*headerList,'---'] # add a '---' line as a delimiter
    with open(currentSessionFilePath, 'w') as f:
       f.writelines(f"{line}\n" for line in lines) 
    return 
  
  def appendToSessionFile(self, textToAppend, sessionFilePath):
    """Append supplied text to the given file, adding a newline at the end
    """
    with open(sessionFilePath, 'a') as f:
       f.write(f"{textToAppend}\n")
    return

  def getListOfRunsFromSessionFile(self, sessionFilePath, delimiterLine='---\n'):
    """Process session file to a list of recordings file names"""
    with open(sessionFilePath,'r') as f:
      lines = f.readlines() # newlines are not removed!
    # Find end of header
    delimIdx = lines.index(delimiterLine)
    # Make list of runs (after header and stripped of newline)
    listOfRuns = [line.rstrip() for line in lines[delimIdx+1:]]
    return listOfRuns

  def constructCurrentSessionFilePath(self, sessionDirectory, sessionFilePrefix, userName):
    timeStamp = time.strftime(r"%Y-%m-%d-%H%M%S")
    currentSessionFilePath = os.path.join(sessionDirectory, f"{sessionFilePrefix}{userName.replace(' ','_')}-{timeStamp}.txt")
    return currentSessionFilePath

  
  def processTrackerFileToRuns(self, mhaFile, segmentationNode, entryRegionName='entryZone', deeperRegionName='deeperZone'):
      raise(Exception("don't use, functionality transferred to Recording objects"))
      """Inputs are name/path to saved tracker recording file, segmentation node with segments 
      an entry zone and a deeper zone (used to define what counts as a run and how they should 
      trimmed). 
      Created during processing: Loaded path as markups curve node, trimmed paths (runs) 
      """
      timeStamps, headSensorTransforms, scopeSensorTransforms = self.import_tracker_recording(mhaFile)
      # Set up the hierarchy order 
      #transformsList = self.gatherTestingTransforms()
      leafNodeName = DEFAULT_LEAF_TRANSFORM_NODE_NAME # TODO: don't hard code this!!
      leafTransformNode = slicer.util.getNode(leafNodeName)
      transformNames, transformsList = self.gatherTransformsFromTransformHierarchy(leafTransformNode)
      # TODO: Verify that transformNames[1] looks like it's the dynamic head sensor and [2] looks like the dynamic scope sensor
      # Replace the single transform matrix arrays in transformsList with the full set from 
      # the loaded file for both the head sensor and scope sensor
      transformsList[HEAD_SENSOR_TRANSFORM_POSITION_IN_HIERARCHY] = headSensorTransforms
      transformsList[SCOPE_SENSOR_TRANSFORM_POSITION_IN_HIERARCHY] = scopeSensorTransforms
      # Get sequence of locations in RAS space
      positions, orientations = self.positions_from_transform_hierarchy(transformsList)
      rawPathModelNode = None #self.modelNodeFromPositions(positions)
      #rawCurveNode = self.markupsCurveFromPositions(positions)
      # Gather runs data
      runsData = self.identifyTrackingRunsFromRawPath(positions, segmentationNode)
      pathRunsModelNodes = []
      for runData in runsData:
          startIdx, endIdx = runData
          runPositions = positions[startIdx:(endIdx+1),:] # NOTE: NOT a deep copy
          runOrientations = orientations[startIdx:(endIdx+1),:]
          #self.trimPathToRange(positions, startIdx, endIdx)
          pathRunModelNode = self.modelNodeFromPositionsAndOrientations(runPositions, runOrientations, scalars=None, sizeFactor=2.0)
          pathRunsModelNodes.append(pathRunModelNode)
      return pathRunsModelNodes, rawPathModelNode

  def modelNodeFromPositionsAndOrientations(self, positions, orientations, scalars=None, sizeFactor=2.0):
      """Create model node with polydata (for non-interactive version of markups)
      Helpful links: 
      https://www.dillonbhuff.com/?p=540
      https://vtk.org/doc/release/5.0/html/a01880.html#:~:text=vtkPolyData%20is%20a%20data%20object,also%20are%20represented.
      """
      # TODO: Add velocities (size), deviations from expert (color)?
      import numpy as np
      pointsArray = positions # an nx3 numpy array
      numPoints = pointsArray.shape[0]
      #diffs = np.diff(pointsArray, axis=0)
      # Expand scalars to array if needed 
      if scalars is None:
        scalars = np.ones(numPoints)
      elif isinstance(scalars, (list, tuple, np.ndarray)) and len(scalars)==1:
        scalars = scalars[0] * np.ones(numPoints)
      else:
        # Single value not in a list
        scalars = scalars * np.ones(numPoints)

      # Create VTK arrays needed for model node
      points = vtk.vtkPoints() # actual pointData locations
      vertices = vtk.vtkCellArray() # handles vertex locations (in our case this will just be all the points, but note that
      # points could be a superset of vertices because points could include the endpoints of lines or corners of polygons 
      # which don't themselves have to be in the list of vertices.  Vertices are what the vtkGlyph3D filter operates on
      lines = vtk.vtkCellArray() # handles lines (and same type would handle polygons if those were being used)

      

      vectors = vtk.vtkFloatArray() # this will affect glyph orientation and is going to be set to the vector to the next point
      vectors.SetNumberOfComponents(3)
      vectors.SetName("Directions")

      sizes = vtk.vtkFloatArray() # this will go in scalars 
      sizes.SetName("Sizes")
      #colors = vtk.vtkFloatArray()
      #colors.SetName("Colors"

      # Assemble arrays of values
      for point, orientation, scalar in zip(positions,orientations,scalars):
        pointID = points.InsertNextPoint(point)
        # Vertices
        cellID = vertices.InsertNextCell(1) # allocates a next cell with space for one point ID (lines would have 2, triangles 3, polygons N)
        vertices.InsertCellPoint(pointID) # fills the first (and only) slot for this cell
        # Vectors
        vectorID = vectors.InsertNextTuple(orientation)
        # Speed?? Could be calculated here and used to size the cones? TODO
        sizes.InsertNextValue(scalar * sizeFactor)
        if pointID != (numPoints-1):
          # Add a line unless this is the very last point
          lines.InsertNextCell(2)
          # allocates a next cell with space for two pointIDs (the endpoints of the line)
          lines.InsertCellPoint(pointID)
          lines.InsertCellPoint(pointID+1)
          
          #size = e[i] * sizeFactor
          #_ = sizes.InsertNextValue(size)
          #colorIdx = cmapIndices[i]
          #_ = colors.InsertNextValue(colorIdx)
          
      ## Create the vtkPolyData
      pointsPolyData = vtk.vtkPolyData()
      pointsPolyData.SetPoints(points)
      pointsPolyData.SetVerts(vertices)
      pointsPolyData.SetLines(lines)
      pointsData = pointsPolyData.GetPointData()
      _ = pointsData.SetScalars(sizes)  #scalars are literally used as size scale factors, I think
      _ = pointsData.SetVectors(vectors)
      #_ = pointsData.AddArray(colors)

      sphere = vtk.vtkSphereSource()  # ConeSource()
      cone = vtk.vtkConeSource()
      cone.SetResolution(18)

      linesPolyData = vtk.vtkPolyData()
      linesPolyData.SetPoints(points)
      linesPolyData.SetLines(lines)

      tubeFilter = vtk.vtkTubeFilter()
      tubeFilter.SetInputData(linesPolyData)
      tubeFilter.SetRadius(0.5)
      tubeFilter.SetNumberOfSides(15)

      glyphFilter = vtk.vtkGlyph3D()
      glyphFilter.SetSourceConnection(cone.GetOutputPort())
      glyphFilter.SetInputData(pointsPolyData)

      modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", 'OriCones')
      modelNode.CreateDefaultDisplayNodes()
      modelDisplay = modelNode.GetDisplayNode()
      # modelDisplay.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileViridis.txt')
      #modelDisplay.SetAndObserveColorNodeID(
      #    "vtkMRMLColorTableNodeFileColdToHotRainbow.txt"
      #)
      # Color table names can be found at https://apidocs.slicer.org/master/classvtkMRMLColorLogic.html
      # I found that changing the color using the Model display node GUI crashes slicer, I don't know why
      #modelDisplay.SetScalarVisibility(True)
      #modelDisplay.SetActiveScalarName("Colors")

      # Connect to glyph output
      modelNode.SetPolyDataConnection(glyphFilter.GetOutputPort())

      # Lines version
      modelNode2 = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'Tube')
      modelNode2.CreateDefaultDisplayNodes()
      modelNode2.SetPolyDataConnection(tubeFilter.GetOutputPort())
      return modelNode, modelNode2

  def modelNodeFromPositions(self, positions):
      """Create model node with polydata (for non-interactive version of markups)
      Helpful links: 
      https://www.dillonbhuff.com/?p=540
      https://vtk.org/doc/release/5.0/html/a01880.html#:~:text=vtkPolyData%20is%20a%20data%20object,also%20are%20represented.
      """
      import numpy as np
      # Create VTK arrays needed for model node
      points = vtk.vtkPoints() # actual pointData locations
      vertices = vtk.vtkCellArray() # handles vertex locations (in our case this will just be all the points, but note that
      # points could be a superset of vertices because points could include the endpoints of lines or corners of polygons 
      # which don't themselves have to be in the list of vertices.  Vertices are what the vtkGlyph3D filter operates on
      lines = vtk.vtkCellArray() # handles lines (and same type would handle polygons if those were being used)

      pointsArray = positions # an nx3 numpy array
      diffs = np.diff(pointsArray, axis=0)

      vectors = vtk.vtkFloatArray() # this will affect glyph orientation and is going to be set to the vector to the next point
      vectors.SetNumberOfComponents(3)
      vectors.SetName("Directions")

      #sizes = vtk.vtkFloatArray() # this will go in scalars 
      #sizes.SetName("Sizes")
      #colors = vtk.vtkFloatArray()
      #colors.SetName("Colors")
      vectors = vtk.vtkFloatArray() # this will affect glyph orientation and is going to be set to the vector to the next point
      vectors.SetNumberOfComponents(3)
      vectors.SetName("Directions")

      # Assemble arrays of values
      for i, vectorThisToNext in enumerate(diffs):
          point = pointsArray[i]
          pointID = points.InsertNextPoint(point)
          # Vertices
          cellID = vertices.InsertNextCell(1) # allocates a next cell with space for one point ID (lines would have 2, triangles 3, polygons N)
          vertices.InsertCellPoint(pointID) # fills the first (and only) slot for this cell
          #size = e[i] * sizeFactor
          #_ = sizes.InsertNextValue(size)
          #colorIdx = cmapIndices[i]
          #_ = colors.InsertNextValue(colorIdx)
          # Vectors
          _ = vectors.InsertNextTuple(vectorThisToNext)
          # Lines
          lines.InsertNextCell(2) # allocates a next cell with space for two pointIDs (the endpoints of the line)
          lines.InsertCellPoint(pointID)
          lines.InsertCellPoint(pointID+1)
      # The above loop leaves off the last contact (because len(diffs) is one less than number of points)
      # Adding this next section to make sure that last point is included
      point = pointsArray[i + 1]
      pointID = points.InsertNextPoint(point)
      vertices.InsertCellPoint(pointID)
      #size = e[i + 1] * sizeFactor
      #_ = sizes.InsertNextValue(size)
      #colorIdx = cmapIndices[i + 1]
      #_ = colors.InsertNextValue(colorIdx)
      _ = vectors.InsertNextTuple(vectorThisToNext)  # just repeat last vector
      # Lines is already complete (there are n-1 lines for n points)

      ## Create the vtkPolyData
      pointsPolyData = vtk.vtkPolyData()
      pointsPolyData.SetPoints(points)
      pointsPolyData.SetVerts(vertices)
      pointsPolyData.SetLines(lines)
      pointsData = pointsPolyData.GetPointData()
      #_ = pointsData.SetScalars(sizes)
      _ = pointsData.SetVectors(vectors)
      #_ = pointsData.AddArray(colors)

      sphere = vtk.vtkSphereSource()  # ConeSource()
      cone = vtk.vtkConeSource()
      cone.SetResolution(18)

      linesPolyData = vtk.vtkPolyData()
      linesPolyData.SetPoints(points)
      linesPolyData.SetLines(lines)

      tubeFilter = vtk.vtkTubeFilter()
      tubeFilter.SetInputData(linesPolyData)
      tubeFilter.SetRadius(1)
      tubeFilter.SetNumberOfSides(15)

      glyphFilter = vtk.vtkGlyph3D()
      glyphFilter.SetSourceConnection(cone.GetOutputPort())
      glyphFilter.SetInputData(pointsPolyData)

      modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", 'Cones')
      modelNode.CreateDefaultDisplayNodes()
      modelDisplay = modelNode.GetDisplayNode()
      # modelDisplay.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileViridis.txt')
      #modelDisplay.SetAndObserveColorNodeID(
      #    "vtkMRMLColorTableNodeFileColdToHotRainbow.txt"
      #)
      # Color table names can be found at https://apidocs.slicer.org/master/classvtkMRMLColorLogic.html
      # I found that changing the color using the Model display node GUI crashes slicer, I don't know why
      #modelDisplay.SetScalarVisibility(True)
      #modelDisplay.SetActiveScalarName("Colors")

      # Connect to glyph output
      modelNode.SetPolyDataConnection(glyphFilter.GetOutputPort())

      # Lines version
      modelNode2 = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', 'Tube')
      modelNode2.CreateDefaultDisplayNodes()
      modelNode2.SetPolyDataConnection(tubeFilter.GetOutputPort())
      return modelNode, modelNode2


  def trimPathToRange(self, markupsNode, startIdx, endIdx, outputMarkupsNode=None):
      """
      """
      import numpy as np
      if outputMarkupsNode is None:
          outName = slicer.mrmlScene.GenerateUniqueName(markupsNode.GetName()+'_trimmed')
          outputMarkupsNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsCurveNode',outName)
      arr = slicer.util.arrayFromMarkupsControlPoints(markupsNode)
      slicer.util.updateMarkupsControlPointsFromArray(outputMarkupsNode, arr[startIdx:(endIdx+1),:])
      return outputMarkupsNode


  def identifyTrackingRunsFromRawPath(self, positionsArray, segmentationNode, entryRegionName='entryZone', deeperRegionName='deeperZone'):
      raise(Exception('accessed ExampleGuidelet.identifyTrackingRunsFromRawPath, do not do that'))
      """ This function should take a markupsNode and process it's control points to trim out unnecessary
      points from before entering the nose and after exiting the nose.  To count as a run, perhaps it should 
      have points both in an "entry" region, and a "deeper" region.  Typical runs would start outside the 
      "entry" zone, enter the entry zone, move onto the "deeper" zone, and finally move back through the "entry" 
      zone and out.  However, it should also handle the cases where recording starts already in the "deeper" 
      region. A run start has the following characteristics:
      1. The first recorded point if it is in the deeperZone.  OR The first recorded point which is in the entryZone 
      and which is followed a continuous series of points which are in the entryZone and which is then followed by
      at least one point in the deeperZone before any points which are neither in the entryZone nor the deeperZone.
      A run ends at the first point after a run has started which is outside the entryZone and deeperZone.  After a 
      run is identified and ended, another run may be present.  Notes should announce when there are events when .
      Another way to think about this would be that every deeperZone point should be part of run, and that run should 
      include any leading or trailing entryZone points connected to it. 
      A list of "run" groupings should be returned.  A run grouping is a list which has information on all the control
      point loctations in the run, along with a matching list of indices into the original control points (to allow 
      time-stamp recovery)
      TODO: Add error correction for aberrant points.  How about, if there is a gap between runs of 
      less than 5 points, then they should just be merged into 
      """
      import numpy as np
      runsData = []
      #points = slicer.util.arrayFromMarkupsControlPoints(markupsNode)
      points = positionsArray
      nPoints = points.shape[0]
      segmentNames = self.getSegmentNamesAtRasPoint(segmentationNode, points)
      # Make a mask of deeperZone points and entryZone points
      deepMask = np.zeros((nPoints))
      entryMask = np.zeros((nPoints))
      for idx, point in enumerate(points):
          segNamesNow = segmentNames[idx]
          deepMask[idx] = 1 if deeperRegionName in segNamesNow else 0
          entryMask[idx] = 1 if entryRegionName in segNamesNow else 0
      
      # Make a mask of entryZone points
      # Identify the first deeperZone point
      # If no deeperZone points, no runs
      if np.all(deepMask==0):
          logging.info(f'No runs present because no points are inside {deeperRegionName} segment!')
          return []
      
      while np.any(deepMask==1):
          # Find first deep point index
          firstDeepIdx = np.argmax(deepMask)
          # Run backwards to find the first contiguous point which is still in either entry or deeper zone
          startIdx = firstDeepIdx
          while True:
              if startIdx==0 or ((deepMask[startIdx-1] != 1) and (entryMask[startIdx-1] != 1)):
                  # This startIdx is the final one
                  break
              else:
                  # Decrement 
                  startIdx = startIdx - 1
          # Run forwards to find the last contiguous point which is still in either entry or deeper zone
          lastIdx = firstDeepIdx
          while True:
              if lastIdx==len(deepMask)-1 or ((deepMask[lastIdx+1] != 1) and (entryMask[lastIdx+1] != 1)):
                  break
              else: 
                  # increment
                  lastIdx = lastIdx + 1
          # Store data
          runData = [startIdx, lastIdx]
          logging.debug(f'Run identified from index {startIdx} to {lastIdx}.')
          runsData.append(runData)
          # Clear this run from deepMask
          deepMask[startIdx:(lastIdx+1)] = 0
      return runsData

 

  def mhaTesting(self, mhaFile=None):
      if mhaFile is None:
          #mhaFile = r'C:\Users\mikeb\Downloads\ExampleGuideletRec-20230601-113809.mhd'
          mhaFile = r"C:/Users/mikeb/Downloads/MayaTanyaRunsData/MayaTanyaRunsData/ExampleGuideletRec-20230602-083158.mhd"
          #mhaFile = r"C:/Users/mikeb/Downloads/MayaTanyaRunsData/MayaTanyaRunsData/ExampleGuideletRec-20230602-073923.mhd"
          #mhaFile = r'C:\Users\mikeb\Downloads\ExampleGuideletRec-20230602-080018.mhd'
          #mhaFile = r'C:/Users/mikeb/Downloads/Recording.igs20230524_103059.mha'
      timeStamps, headSensorTransforms, scopeSensorTransforms = self.import_tracker_recording(mhaFile)
      # Set up the hierarchy order 
      transformsList = self.gatherTestingTransforms()
      transformsList[1] = headSensorTransforms
      transformsList[2] = scopeSensorTransforms
      positions = self.positions_from_transform_hierarchy(transformsList)
      curveNode = self.markupsCurveFromPositions(positions)

      return timeStamps, curveNode, positions, headSensorTransforms, scopeSensorTransforms
  
  def gatherTransformsFromTransformHierarchy(self, leafTransformNode):
    raise(Exception('Accessed ExampleGuidelet.gatherTransformsFromTransformHierarchy(), use recordings version instead'))
    """ Use the existing scene transform hierarchy to build transformList
    """
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    leafTransformNode.GetTransformNodeID()
    curT = leafTransformNode
    transformNodeList = [leafTransformNode]
    while curT.GetTransformNodeID():
      # Get parent transform node
      curT = slicer.mrmlScene.GetNodeByID(curT.GetTransformNodeID())
      transformNodeList.append(curT)
    # Reverse order so decending hierarchy rather than ascending
    transformNodeList.reverse()
    transformNames = [tNode.GetName() for tNode in transformNodeList]
    transformList = [slicer.util.arrayFromTransformMatrix(tNode) for tNode in transformNodeList]
    return transformNames, transformList

  def gatherTestingTransforms(self):
      """Requires tracking example 2 scene loaded"""
      """ 
      HeadSensorToHeadSTL 
      -> EmTrackerToHeadSenso (tracked) 
          -> StylusSensorToEmTrac (tracked)
            -> NeedleTipToStylusSen (currently just rotation of axes)
                -> extra (to allow for adjustment of tip translation between sensor and camera)
                  -> NeedleModel
      """
      from slicer.util import getNode
      import numpy as np
      headSToStlNode = getNode('HeadSensorToHeadSTL')
      TrToHeadS = getNode('EmTrackerToHeadSenso')
      ScopeSToTr = getNode('StylusSensorToEmTrac')
      ScopeTipToScopeS = getNode('NeedleTipToStylusSen')
      ExtraTipAdjustment = getNode('extra')
      TNodeList = [headSToStlNode, TrToHeadS, ScopeSToTr, ScopeTipToScopeS, ExtraTipAdjustment]
      transformList = [slicer.util.arrayFromTransformMatrix(TNode) for TNode in TNodeList]
      return transformList


  def markupsCurveFromPositions(self, positions):
      """Create markupsCurveNode from Nx3 numpy array"""
      pathName = slicer.mrmlScene.GenerateUniqueName('Path')
      curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", pathName)
      slicer.util.updateMarkupsControlPointsFromArray(curveNode, positions)
      return curveNode


  def positions_from_transform_hierarchy(self, transformsList):
      """Compute series of locations given a list of hierarchical transform matrices. 
      transformsList[i] must either be a single 4x4 array or a 4x4xN array, where N is 
      the number of time step frames. transformsList[i] is the parent transform to 
      transformsList[i+1]
      """
      import numpy as np
      
      numFramesPer = np.zeros((len(transformsList)),dtype=int)
      for idx, tListItem in enumerate(transformsList):
          numDims = tListItem.ndim
          if numDims==3:
              numFramesPer[idx] = tListItem.shape[2]
          elif numDims==2:
              numFramesPer[idx] = 1
      # NumFramesPer elements should now all either equal 1 or the same number of frames
      numFrames = np.max(numFramesPer)
      assert np.all((numFramesPer==1) | (numFramesPer==numFrames) ), 'All transforms supplied must have either a single frame or the same number of frames!'        
      # Calculate positions from the sequence of transforms
      positions = np.zeros((numFrames, 3))
      orientations = np.zeros((numFrames,3))
      origin = np.zeros((4))
      origin[3] = 1 # homogenous coordinate for a point
      direction = np.zeros((4))
      direction[2] = 1 # [0,0,1,0]
      direction[3] = 0 # homogenous coord for a vector
      concatTransforms = []
      for frameNum in range(numFrames):
          # Assemble the correct list of transforms for this frame
          currentTransformList = []
          for listIdx in range(len(transformsList)):
              if numFramesPer[listIdx]==1:
                  curTransform = transformsList[listIdx]
              else:
                  curTransform = transformsList[listIdx][:, :, frameNum]
              currentTransformList.append(curTransform)
          # Apply tranforms in order to origin position
          concatTransform = np.linalg.multi_dot(currentTransformList)
          concatTransforms.append(concatTransform)
          currentPosition4 = concatTransform @ origin
          currentOrientation4 = concatTransform @ direction
          positions[frameNum,:] = currentPosition4[0:3]
          orientations[frameNum,:] = currentOrientation4[0:3]
      return positions, orientations


  def import_tracker_recording(self, mha_file_path):
      raise(Exception('Accessed ExampleGuidelet.import_tracker_recording(). Use Recording class version instead!'))
      """Import the sequence of transforms stored in one of the guidelet mhd files.
      """
      import numpy as np
      import re
      logging.debug(f'Opening file: {mha_file_path} ...')
      # Just parse far enough to get number of timesteps
      numSeqFrames = 0
      with open(mha_file_path) as f:
          for line in f:
              if line.startswith('DimSize'):
                  numSeqFrames = int(line.rstrip().split()[-1])
                  logging.debug(f'Found DimSize line: {numSeqFrames} time steps in file')
                  break
      if numSeqFrames==0:
          raise(Exception('DimSize line not found in mha file processing, cannot determine number of processed time steps.'))
      # Preallocate transform arrays
      timeStamps = np.zeros((numSeqFrames))
      headSensorTransforms = np.zeros((4,4,numSeqFrames))
      scopeSensorTransforms = np.zeros((4,4,numSeqFrames))
      # Sequence line info pattern
      transformLinePatt = re.compile(r'Seq_Frame(?P<frameNumber>\d\d\d\d)_(?P<transformName>\w+) =(?P<matrix>( -?\d+(\.\d+)?([Ee][+-]?\d+)?){16})')
      timeStampLinePatt = re.compile(r'Seq_Frame(?P<frameNumber>\d\d\d\d)_Timestamp = (?P<timeStamp>-?\d+(\.\d+)?)')
      # Read the file line by line
      with open(mha_file_path) as f:
          keepgoing = True
          for line in f:
              if m := transformLinePatt.search(line):
                  # Process the result
                  groupDict = m.groupdict()
                  #logging.debug(groupDict.__str__())
                  frameNum = int(groupDict['frameNumber'])
                  transformName = groupDict['transformName']
                  transformMatrixList = groupDict['matrix'].lstrip().split() # row order 
                  transformMatrix = np.array(transformMatrixList).reshape((4,4))
                  # Store in arrays
                  if re.search('Head', transformName):
                      headSensorTransforms[:,:,frameNum] = np.linalg.inv(transformMatrix.astype('float64'))  # THESE NEED TO BE INVERTED!!!
                  elif re.search('Stylus', transformName):
                      scopeSensorTransforms[:,:,frameNum] = transformMatrix
              if m := timeStampLinePatt.search(line):
                  groupDict = m.groupdict()
                  frameNum = int(groupDict['frameNumber'])
                  timeStamp = float(groupDict['timeStamp'])
                  timeStamps[frameNum] = timeStamp
      return timeStamps, headSensorTransforms, scopeSensorTransforms

  def createVolumeFromROIandVoxelSize(
        self, ROINode, voxelSizeMm=[1.0, 1.0, 1.0], prioritizeVoxelSize=True
    ):
        """Create an empty scalar volume node with the given resolution, location, and
        orientation. The resolution must be given directly (single or scalar value interpreted
        as an isotropic edge length), and the location, size, and orientation are derived from
        the ROINode (a vtkMRMLAnnotationROINode). If prioritizeVoxelSize is True (the default),
        and the size of the ROI is not already an integer number of voxels across in each dimension,
        the ROI is minimally expanded to the next integer number of voxels across in each dimension.
        If prioritizeVoxelSize is False, then the ROI is left unchanged, and the voxel dimensions
        are minimally adjusted such that the existing ROI is an integer number of voxels across.
        """
        import numpy as np

        # Ensure resolutionMm can be converted to a list of 3 voxel edge lengths
        # If voxel size is a scalar or a one-element list, interpret that as a request for
        # isotropic voxels with that edge length
        if hasattr(voxelSizeMm, "__len__"):
            if len(voxelSizeMm) == 1:
                voxelSizeMm = [voxelSizeMm[0]] * 3
            elif not len(voxelSizeMm) == 3:
                raise Exception(
                    "voxelSizeMm must either have one or 3 elements; it does not."
                )
        else:
            try:
                v = float(voxelSizeMm)
                voxelSizeMm = [v] * 3
            except:
                raise Exception(
                    "voxelSizeMm does not appear to be a number or a list of one or three numbers."
                )

        # Resolve any tension between the ROI size and resolution if ROI is not an integer number of voxels in all dimensions
        ROIRadiusXYZMm = [0] * 3  # initialize
        ROINode.GetRadiusXYZ(ROIRadiusXYZMm)  # fill in ROI sizes
        ROIDiamXYZMm = 2 * np.array(
            ROIRadiusXYZMm
        )  # need to double radii to get box dims
        numVoxelsAcrossFloat = np.divide(ROIDiamXYZMm, voxelSizeMm)
        voxelTol = 0.1  # fraction of a voxel it is OK to shrink the ROI by (rather than growing by 1-voxelTol voxels)
        if prioritizeVoxelSize:
            # Adjust ROI size by increasing it to the next integer multiple of the voxel edge length
            numVoxAcrossInt = []
            for voxAcross in numVoxelsAcrossFloat:
                # If over by less voxelTol of a voxel, don't ceiling it
                diff = voxAcross - np.round(voxAcross)
                if diff > 0 and diff < voxelTol:
                    voxAcrossInt = np.round(
                        voxAcross
                    )  # round it down, which will shrink the ROI by up to voxelTol voxels
                else:
                    voxAcrossInt = np.ceil(
                        voxAcross
                    )  # otherwise, grow ROI to the next integer voxel size
                numVoxAcrossInt.append(voxAcrossInt)
            # Figure out new ROI dimensions
            adjustedROIDiamXYZMm = np.multiply(numVoxAcrossInt, voxelSizeMm)
            adjustedROIRadiusXYZMm = (
                0.5 * adjustedROIDiamXYZMm
            )  # radii are half box dims
            # Apply adjustment
            ROINode.SetRadiusXYZ(adjustedROIRadiusXYZMm)
        else:  # prioritize ROI dimension, adjust voxel resolution
            numVoxAcrossInt = np.round(numVoxelsAcrossFloat)
            # Adjust voxel resolution
            adjustedVoxelSizeMm = np.divide(ROIDiamXYZMm, numVoxAcrossInt)
            voxelSizeMm = adjustedVoxelSizeMm

        #
        volumeName = "OutputTemplateVolume"
        voxelType = (
            vtk.VTK_UNSIGNED_INT
        )  # not sure if this locks in anything for resampling, if so, might be an issue
        imageDirections, origin = self.getROIDirectionsAndOrigin(
            ROINode
        )  # these are currently not normalized!

        # Create volume node
        templateVolNode = self.createVolumeNodeFromScratch(
            volumeName,
            imageSizeVox=numVoxAcrossInt,
            imageOrigin=origin,
            imageSpacingMm=voxelSizeMm,
            imageDirections=imageDirections,
            voxelType=voxelType,
        )
        return templateVolNode

  def createVolumeNodeFromScratch(
      self,
      nodeName="VolumeFromScratch",
      imageSizeVox=[256, 256, 256],  # image size in voxels
      imageSpacingMm=[2.0, 2.0, 2.0],  # voxel size in mm
      imageOrigin=[0.0, 0.0, 0.0],
      imageDirections=[
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
      ],  # Image axis directions IJK to RAS,  (these should be orthogonal!)
      fillVoxelValue=0,
      voxelType=vtk.VTK_UNSIGNED_CHAR,
  ):
      """Create a scalar volume node from scratch, given information on"""
      imageData = vtk.vtkImageData()
      imageSizeVoxInt = [int(v) for v in imageSizeVox]
      imageData.SetDimensions(imageSizeVoxInt)
      imageData.AllocateScalars(voxelType, 1)
      imageData.GetPointData().GetScalars().Fill(fillVoxelValue)
      # Normalize and check orthogonality image directions
      import numpy as np
      import logging

      imageDirectionsUnit = [np.divide(d, np.linalg.norm(d)) for d in imageDirections]
      angleTolDegrees = 1  # allow non-orthogonality up to 1 degree
      for pair in ([0, 1], [1, 2], [2, 0]):
          angleBetween = np.degrees(
              np.arccos(
                  np.dot(imageDirectionsUnit[pair[0]], imageDirectionsUnit[pair[1]])
              )
          )
          if abs(90 - angleBetween) > angleTolDegrees:
              logging.warning(
                  "Warning! imageDirections #%i and #%i supplied to createVolumeNodeFromScratch are not orthogonal!"
                  % (pair[0], pair[1])
              )
              # Continue anyway, because volume nodes can sort of handle non-orthogonal image directions (though they're not generally expected)
      # Create volume node
      volumeNode = slicer.mrmlScene.AddNewNodeByClass(
          "vtkMRMLScalarVolumeNode", nodeName
      )
      volumeNode.SetOrigin(imageOrigin)
      volumeNode.SetSpacing(imageSpacingMm)
      volumeNode.SetIJKToRASDirections(imageDirections)
      volumeNode.SetAndObserveImageData(imageData)
      volumeNode.CreateDefaultDisplayNodes()
      volumeNode.CreateDefaultStorageNode()
      return volumeNode

  # ROINode.GetControlPointWorldCoordinates(0,p) puts transformed center point coordinates into p
  # p = ROINode.GetXYZ() puts untransformed center point coordinates into p
  # Interactive modification of ROI modifies the ROINode like SetXYZ and SetRadiusXYZ, does not affect transform
  # Orientation must come from transform.  N.B. transform node is around 0,0,0, NOT around ROI XYZ center. If there
  # is scaling in the transform, the GetXYZ() coord is scaled by the transform. That is [5,0,0] with a transform
  # that doubles R and moves it by 20 moves the CENTER of the drawn ROI
  # ROINode.GetTransformNodeID() allows retrieval of transform.  Let's assume for now that the transform node does
  # not involve any scaling (actually, we could check with decomp and throw a warning if it does)
  # Also, identified a problem with CropVolume module, where scaling in the transform leads to cropped volume not
  # matching ROI (display box is scaled, volume center is not?)

  def getROIDirectionsAndOrigin(self, roiNode):
      import numpy as np

      # Processing is different depending on whether the roiNode is AnnotationsMarkup or MarkupsROINode
      if isinstance(roiNode, slicer.vtkMRMLMarkupsROINode):
          axis0 = [0, 0, 0]
          roiNode.GetXAxisWorld(
              axis0
          )  # This respects soft transforms applied to the ROI!
          axis1 = [0, 0, 0]
          roiNode.GetYAxisWorld(axis1)
          axis2 = [0, 0, 0]
          roiNode.GetZAxisWorld(axis2)
          # These axes are the columns of the IJKToRAS directions matrix, but when
          # we supply a list of directions to the imageDirections, it takes a list of rows,
          # so we need to transpose
          directions = np.transpose(
              np.stack((axis0, axis1, axis2))
          )  # for imageDirections
          center = [0, 0, 0]
          roiNode.GetCenterWorld(center)
          radiusXYZ = [0, 0, 0]
          roiNode.GetRadiusXYZ(radiusXYZ)
          # The origin in the corner where the axes all point along the ROI
          origin = (
              np.array(center)
              - np.array(axis0) * radiusXYZ[0]
              - np.array(axis1) * radiusXYZ[1]
              - np.array(axis2) * radiusXYZ[2]
          )
      else:
          # Input is not markupsROINode, must be older annotations ROI instead
          T_id = roiNode.GetTransformNodeID()
          if T_id:
              T = slicer.mrmlScene.GetNodeByID(T_id)
          else:
              T = None
          if T:
              # Transform node is present
              # transformMatrix = slicer.util.arrayFromTransformMatrix(T) # numpy 4x4 array
              # if nested transform, then above will fail! # TODO TODO
              worldToROITransformMatrix = vtk.vtkMatrix4x4()
              T.GetMatrixTransformBetweenNodes(None, T, worldToROITransformMatrix)
              # then convert to numpy
          else:
              worldToROITransformMatrix = (
                  vtk.vtkMatrix4x4()
              )  # defaults to identity matrix
              # transformMatrix = np.eye(4)
          # Convert to directions (for image directions)
          axis0 = np.array(
              [worldToROITransformMatrix.GetElement(i, 0) for i in range(3)]
          )
          axis1 = np.array(
              [worldToROITransformMatrix.GetElement(i, 1) for i in range(3)]
          )
          axis2 = np.array(
              [worldToROITransformMatrix.GetElement(i, 2) for i in range(3)]
          )
          directions = (axis0, axis1, axis2)  # for imageDirections
          # Find origin of roiNode (RAS world coord)
          # Origin is Center - radius1 * direction1 - radius2 * direction2 - radius3 * direction3
          ROIToWorldTransformMatrix = vtk.vtkMatrix4x4()
          ROIToWorldTransformMatrix.DeepCopy(worldToROITransformMatrix)  # copy
          ROIToWorldTransformMatrix.Invert()  # invert worldToROI to get ROIToWorld
          # To adjust the origin location I need to use the axes of the ROIToWorldTransformMatrix
          ax0 = np.array(
              [ROIToWorldTransformMatrix.GetElement(i, 0) for i in range(3)]
          )
          ax1 = np.array(
              [ROIToWorldTransformMatrix.GetElement(i, 1) for i in range(3)]
          )
          ax2 = np.array(
              [ROIToWorldTransformMatrix.GetElement(i, 2) for i in range(3)]
          )
          boxDirections = (ax0, ax1, ax2)
          TransformOrigin4 = ROIToWorldTransformMatrix.MultiplyPoint([0, 0, 0, 1])
          TransformOrigin = TransformOrigin4[:3]
          roiCenter = [0] * 3  # intialize
          roiNode.GetXYZ(roiCenter)  # fill
          # I want to transform the roiCenter using roiToWorld
          transfRoiCenter4 = ROIToWorldTransformMatrix.MultiplyPoint([*roiCenter, 1])
          transfRoiCenter = transfRoiCenter4[:3]
          # Now need to subtract
          radXYZ = [0] * 3
          roiNode.GetRadiusXYZ(radXYZ)
          origin = (
              np.array(transfRoiCenter)
              - ax0 * radXYZ[0]
              - ax1 * radXYZ[1]
              - ax2 * radXYZ[2]
          )

      # Return outputs
      return directions, origin

  def addValuesToDefaultConfiguration(self):
    GuideletLogic.addValuesToDefaultConfiguration(self)
    moduleDir = os.path.dirname(slicer.modules.exampleguidelet.path)
    defaultUserSessionsSavePath = os.path.join(moduleDir, 'UserSessionResults') # TODO: Create folder if it doesn't exist
    defaultSceneSavePath = os.path.join(moduleDir, 'SavedScenes')
    moduleDirectoryPath = slicer.modules.exampleguidelet.path.replace('ExampleGuidelet.py','')
    settingList = {
                   'StyleSheet' : moduleDirectoryPath + 'Resources/StyleSheets/ExampleGuideletStyle.qss',
                   'LiveUltrasoundNodeName': 'Image_Reference',
                   'TestMode' : 'False',
                   'RecordingFilenamePrefix' : 'AirwayTrackerRec-',
                   'UserSessionResultsDirectory': defaultUserSessionsSavePath, # folder to put session files in
                   'SavedScenesDirectory': defaultSceneSavePath, #overwrites the default setting param of base
                   }
    self.updateSettings(settingList, 'Default')


class ExampleGuideletTest(GuideletTest):
  """This is the test case for your scripted module.
  """

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    GuideletTest.runTest(self)
    #self.test_ExampleGuidelet1() #add applet specific tests here


class ExampleGuideletGuidelet(Guidelet):

  def __init__(self, parent, logic, configurationName='Default'):
    #self.calibrationCollapsibleButton = None
    try:
      slicer.modules.plusremote
    except:
      raise Exception('Error: Could not find Plus Remote module. Please install the SlicerOpenIGTLink extension')
    
    self.plusRemoteLogic = slicer.modules.plusremote.logic()
    self.plusRemoteNode = None
    # Set up icon paths
    fileDir = os.path.dirname(__file__)
    iconPathRecord = os.path.join(fileDir, 'Resources', 'Icons', 'icon_Record.png')
    iconPathStop = os.path.join(fileDir, 'Resources', 'Icons', 'icon_Stop.png')

    if os.path.isfile(iconPathRecord):
      self.recordIcon = qt.QIcon(iconPathRecord)
    else:
       logging.warning(f'Icon not found at {iconPathRecord}!')
    if os.path.isfile(iconPathStop):
      self.stopIcon = qt.QIcon(iconPathStop)

    # Init guidelet 
    Guidelet.__init__(self, parent, logic, configurationName)
    self._updatingGuideletGUIFromParameterNode = False
    self.updateParameterNodeFromGuideletGUI() # force initial update from loaded GUI values (could also set up parameter node 
    # ahead of time, but if we don't do either we end up trying to update the GUI from empty parameter node fields)

    logging.debug('ExampleGuideletGuidelet.__init__')

    self.logic.addValuesToDefaultConfiguration()

    moduleDirectoryPath = slicer.modules.exampleguidelet.path.replace('ExampleGuidelet.py', '')

    # Set up main frame.

    self.sliceletDockWidget.setObjectName('ExampleGuideletPanel')
    self.sliceletDockWidget.setWindowTitle('Airway Tracker')
    self.mainWindow.setWindowTitle('Airway Tracker')
    self.mainWindow.windowIcon = qt.QIcon(moduleDirectoryPath + '/Resources/Icons/ExampleGuidelet.png')

    self.setupScene()

    self.navigationView = self.VIEW_3D

    # Setting button open on startup.
    #self.calibrationCollapsibleButton.setProperty('collapsed', False)

  def updateParameterNodeFromGuideletGUI(self, caller=None, event=None):
     """ Update parameter node values from current GUI information """
     self.parameterNode.SetParameter('userNameLineEditString', self.userNameLineEdit.text)
     self.parameterNode.SetParameter('experienceLevelComboBoxCurrentIndex', str(self.experienceLevelComboBox.currentIndex))
     self.parameterNode.SetParameter('experienceLevelComboBoxCurrentString', self.experienceLevelComboBox.currentText)
     self.parameterNode.SetParameter('roleComboBoxCurrentIndex', str(self.roleComboBox.currentIndex))
     self.parameterNode.SetParameter('roleComboBoxCurrentText', self.roleComboBox.currentText)
     self.parameterNode.SetParameter('airwayZoneSegmentationNodeID', self.airwayZoneSegmentationNodeSelector.currentNodeID)
     self.parameterNode.SetParameter('leafTransformNodeID', self.leafTransformNodeSelector.currentNodeID)

  def updateGuideletGUIFromParameterNode(self, caller=None, event=None):
     """ Update Guidelet GUI elements from parameter values"""
     if self.parameterNode is None or self._updatingGuideletGUIFromParameterNode:
        return
     self._updatingGuideletGUIFromParameterNode = True # prevent infinite update loops
     # User name
     self.userNameLineEdit.text = self.parameterNode.GetParameter('userNameLineEditString')
     # Experience level
     self.experienceLevelComboBox.setCurrentIndex(int(self.parameterNode.GetParameter('experienceLevelComboBoxCurrentIndex')))
     # current text is updated automatically with the index update ^^
     # Role
     self.roleComboBox.setCurrentIndex(int(self.parameterNode.GetParameter('roleComboBoxCurrentIndex')))
     # AirwayZone Segmentation
     self.airwayZoneSegmentationNodeSelector.setCurrentNodeID(self.parameterNode.GetParameter('airwayZoneSegmentationNodeID'))
     self.leafTransformNodeSelector.setCurrentNodeID(self.parameterNode.GetParameter('leafTransformNodeID'))
     # List of runs
     # List of expert runs
     # NOTE: Parameter node settings related to the Advanced panel are handled in Guidelet.py
     self._updatingGuideletGUIFromParameterNode = False


  def createFeaturePanels(self):
    # Create GUI panels.

    self.calibrationCollapsibleButton = ctk.ctkCollapsibleButton()
    self.patientSetupPanel()

    featurePanelList = Guidelet.createFeaturePanels(self)

    featurePanelList[len(featurePanelList):] = [self.calibrationCollapsibleButton]

    return featurePanelList


  def __del__(self):#common
    self.preCleanup()


  # Clean up when guidelet is closed
  def preCleanup(self):#common
    Guidelet.preCleanup(self)
    self.disconnect()

    logging.debug('preCleanup')

  def createPlusConnector(self):
    connectorNode = slicer.mrmlScene.GetFirstNodeByName('PlusConnector')
    if not connectorNode:
      connectorNode = slicer.vtkMRMLIGTLConnectorNode()
      slicer.mrmlScene.AddNode(connectorNode)
      connectorNode.SetName('PlusConnector')
      hostNamePort = self.parameterNode.GetParameter('PlusServerHostNamePort') # example: "localhost:18944"
      [hostName, port] = hostNamePort.split(':')
      connectorNode.SetTypeClient(hostName, int(port))
      logging.debug("PlusConnector created")
    return connectorNode
  
  def onConnectorNodeConnected_Ex(self):
    #self.freezeUltrasoundButton.setText('Freeze')
    self.startStopRecordingButton.setEnabled(True)

  def onConnectorNodeDisconnected_Ex(self):
    #self.freezeUltrasoundButton.setText('Un-freeze')
    if self.parameterNode.GetParameter('RecordingEnabledWhenConnectorNodeDisconnected') == 'False':
      self.startStopRecordingButton.setEnabled(False)

  def setupConnections(self):
    logging.debug('ScoliUs.setupConnections()')
    Guidelet.setupConnections(self)
    self.startStopRecordingButton.connect('clicked(bool)', self.onStartStopRecordingClicked)
    self.userNameLineEdit.connect('editingFinished()', self.updateParameterNodeFromGuideletGUI)
    self.roleComboBox.connect('currentIndexChanged(int)', self.updateParameterNodeFromGuideletGUI)
    self.experienceLevelComboBox.connect('currentIndexChanged(int)', self.updateParameterNodeFromGuideletGUI)
    self.saveUserInfoButton.connect('clicked(bool)', self.saveUserInfoButtonClicked)
    self.displaySelectedRunButton.connect('clicked(bool)', self.onDisplaySelectedRunClicked)

    #self.calibrationCollapsibleButton.connect('toggled(bool)', self.onPatientSetupPanelToggled)
    #self.exampleButton.connect('clicked(bool)', self.onExampleButtonClicked)
    # TODO: Ensure disconnect() has all matching disconnections

  def onDisplaySelectedRunClicked(self):
    """Display the currently selected run in the 3D view """
    runFileName = self.getRunToReviewFileName()
    recordingsDataDirectory = self.parameterNode.GetParameter('PlusAppDataDirectory')
    self.logic.displayRunFromFile()

  def getRunToReviewFileName(self):
    """return the file name of the recording file for the currently selected run in the
    runToReviewComboBox. Will need to be adjusted if a different representation is used
    in the comboBox than the raw file name"""
    return self.runToReviewComboBox.currentText

  def saveUserInfoButtonClicked(self, bool):
    """Update the current user text section and save a session text file"""
    # User
    userText = self.parameterNode.GetParameter('userNameLineEditString')
    if not userText:
      userText = '*no current user saved*'  
    self.parameterNode.SetParameter('CurrentUserText', userText)
    self.currentUserNameLabel.text = userText
    # Experience level
    experienceText = self.parameterNode.GetParameter('experienceLevelComboBoxCurrentString')
    if not experienceText:
      experienceText = '*not set*'
    self.parameterNode.SetParameter('CurrentUserExperienceLevelText', experienceText)
    self.currentExperienceLevelLabel.text = experienceText
    # Role
    roleText = self.parameterNode.GetParameter('roleComboBoxCurrentText')
    if not roleText:
       roleText = '*not set*'
    self.parameterNode.SetParameter('CurrentUserRoleText', roleText)
    self.currentRoleLabel.text = roleText
    # Create Session File to hold results
    sessionDirectory = self.parameterNode.GetParameter('UserSessionResultsDirectory')
    
    userName = self.parameterNode.GetParameter('CurrentUserText')
    userDict = {'userName': userName,
                'experienceLevel': experienceText,
                'role': roleText}
    self.currentSession = Session(userDict)
    self.currentSession.saveToFile(sessionDirectory)
    self.parameterNode.SetParameter('CurrentSessionFilePath', self.currentSession.savedFilePathName)
    # Clear out RunsData for previous session
    listOfRuns = self.logic.getListOfRunsFromSessionFile(self.currentSession.savedFilePathName)
    self.updateRunsToReview(listOfRuns)
    delim = '|' # list delimiter for parameter node lists #TODO: store in to parameter node 
    self.parameterNode.SetParameter('RunsData', delim.join(listOfRuns))

  def onStartStopRecordingClicked(self):
    self.captureDeviceName = self.parameterNode.GetParameter('PLUSCaptureDeviceName')
    if self.startStopRecordingButton.isChecked():
      self.startStopRecordingButton.setText("  Stop Recording")
      self.startStopRecordingButton.setIcon(self.stopIcon)
      self.startStopRecordingButton.setToolTip("Recording is being started...")
      if self.captureDeviceName  != '':
        # Important to save as .mhd because that does not require lengthy finalization (merging into a single file)
        recordPrefix = self.parameterNode.GetParameter('RecordingFilenamePrefix')
        recordExt = self.parameterNode.GetParameter('RecordingFilenameExtension')
        userName = self.parameterNode.GetParameter('CurrentUserText')
        userExp = self.parameterNode.GetParameter('CurrentUserExperienceLevelText')
        userRole = self.parameterNode.GetParameter('CurrentUserRoleText')
        timeStamp = time.strftime(r"%Y-%m-%d-%H%M%S")
        self.recordingFileName = f"{recordPrefix}{userName}-{userExp}-{userRole}-{timeStamp}{recordExt}".replace(' ','_') # replace spaces with underscores
        #self.recordingFileName =  recordPrefix + time.strftime("%Y%m%d-%H%M%S") + recordExt

        logging.info("Starting recording to: {0}".format(self.recordingFileName))

        self.logic.appendToSessionFile(self.recordingFileName, self.currentSession.savedFilePathName)

        self.plusRemoteNode.SetCurrentCaptureID(self.captureDeviceName)
        self.plusRemoteNode.SetRecordingFilename(self.recordingFileName)
        self.plusRemoteLogic.StartRecording(self.plusRemoteNode)

    else:
      self.startStopRecordingButton.setText("  Start Recording")
      self.startStopRecordingButton.setIcon(self.recordIcon)
      self.startStopRecordingButton.setToolTip( "Recording is being stopped..." )
      if self.captureDeviceName  != '':
        logging.info("Stopping recording")
        self.plusRemoteNode.SetCurrentCaptureID(self.captureDeviceName)
        self.plusRemoteLogic.StopRecording(self.plusRemoteNode)
        # Add the new recording to the current session
        recordingsDirectory = self.parameterNode.GetParameter('PlusAppDataDirectory')
        newRecording = Recording(self.currentSession, os.path.join(recordingsDirectory, self.recordingFileName))
        self.currentSession.addRecording(newRecording)
        newRecording.processRecordingToScopeRuns(self.sceneLeafTransformNode, self.airwayZoneSegmentationNode, 'airwayZone')


        # Add the current run to the current dropdown list of RunsData. TODO!
        listOfRecordingFileNames = self.currentSession.getListOfRecordingFileNames()

        # Actually, this ^^ should be the a list of runs, not list of recording file names TODO TODO TODO
        #listOfRuns = self.logic.getListOfRunsFromSessionFile(self.currentSessionFilePath)
        self.updateRunsToReview(listOfRuns)

  def updateRunsToReview(self, listOfRuns):
    """From a list of runs (filenames of recordings), update the dropdown"""
    # Remove all items
    self.runToReviewComboBox.clear()
    if listOfRuns:
      for runFileName in listOfRuns:
        self.runToReviewComboBox.addItem(runFileName)
    else: 
       self.runToReviewComboBox.addItem('*no runs recorded this session*')
    

  def setupScene(self): #applet specific
    logging.debug('ExampleGuideletGuidelet.setupScene')

    '''
    ReferenceToRas transform is used in almost all IGT applications. Reference is the coordinate system
    of a tool fixed to the patient. Tools are tracked relative to Reference, to compensate for patient
    motion. ReferenceToRas makes sure that everything is displayed in an anatomical coordinate system, i.e.
    R, A, and S (Right, Anterior, and Superior) directions in Slicer are correct relative to any
    images or tracked tools displayed.
    ReferenceToRas is needed for initialization, so we need to set it up before calling Guidelet.setupScene().
    '''

    try:
      self.referenceToRas = slicer.util.getNode('EmTrackerToHeadSenso')
    except slicer.util.MRMLNodeNotFoundException:
      self.referenceToRas = None
    ## self.referenceToRas = slicer.util.getNode('ReferenceToRas')
    if not self.referenceToRas:
      self.referenceToRas=slicer.vtkMRMLLinearTransformNode()
      self.referenceToRas.SetName("ReferenceToRas")
      m = self.logic.readTransformFromSettings('ReferenceToRas', self.configurationName)
      if m is None:
        m = self.logic.createMatrixFromString('1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1')
      self.referenceToRas.SetMatrixTransformToParent(m)
      slicer.mrmlScene.AddNode(self.referenceToRas)

    # Guidelet.setupScene(self) # <--connection with Plus server is made here
    # Guidelet.setupScene just calls AirwayTrackerClass.setupScene, which only sets up the plusRemoteNode
    # (and the reslice driver for the ultrasound version, but we're not using that currently)
    # Moving that code here
    logging.debug('ExampleGuideletGuidelet.setupScene: Getting/Creating PlusRemoteNode and observing')
    self.plusRemoteNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLPlusRemoteNode')
    if self.plusRemoteNode is None:
      self.plusRemoteNode = slicer.vtkMRMLPlusRemoteNode()
      self.plusRemoteNode.SetName("PlusRemoteNode")
      slicer.mrmlScene.AddNode(self.plusRemoteNode)
    self.plusRemoteNode.AddObserver(slicer.vtkMRMLPlusRemoteNode.RecordingStartedEvent, self.recordingCommandCompleted)
    self.plusRemoteNode.AddObserver(slicer.vtkMRMLPlusRemoteNode.RecordingCompletedEvent, self.recordingCommandCompleted)
    self.plusRemoteNode.SetAndObserveOpenIGTLinkConnectorNode(self.connectorNode) 


    # Not sure why 'EmTrackerToHeadSenso' didn't exist yet, trying processing events here
    slicer.app.processEvents() 

    # Load airwayZone segmentation
    airwayZoneSegmentationNode = slicer.util.loadSegmentation(DEFAULT_AIRWAYZONE_SEGMENTATION)
    self.parameterNode.SetParameter('airwayZoneSegmentationNodeID', airwayZoneSegmentationNode.GetID())
    self.airwayZoneSegmentationNodeSelector.setCurrentNodeID(airwayZoneSegmentationNode.GetID())
    

    # Use PegNeckHead?
    usingPegNeckHead = True
    slicer.app.processEvents()
    # Hide slice view annotations (patient name, scale, color bar, etc.) as they
    # decrease reslicing performance by 20%-100%
    logging.debug('Hide slice view annotations')
    import DataProbe
    dataProbeUtil=DataProbe.DataProbeLib.DataProbeUtil()
    dataProbeParameterNode=dataProbeUtil.getParameterNode()
    dataProbeParameterNode.SetParameter('showSliceViewAnnotations', '0')

    # Transforms

    logging.debug('Gather transforms')

    # Check if expected transforms are available
    try:
       self.EmTrackerToHeadSensor = slicer.util.getNode('EmTrackerToHeadSenso')
    except slicer.util.MRMLNodeNotFoundException:
       # Conclude we are in testing mode for now
       slicer.util.errorDisplay("Expected transform not found, running it test/debug mode!")
       return # return early since the rest of the method will fail

    self.EmTrackerToHeadSensor = slicer.util.getNode('EmTrackerToHeadSenso')
    self.StylusSensorToEmTracker = slicer.util.getNode('StylusSensorToEmTrac')
    self.StylusTipToStylusSensor = slicer.util.getNode('StylusTipToStylusSen')
    self.NeedleTipToStylusSensor = slicer.util.getNode('NeedleTipToStylusSen')
    if usingPegNeckHead:
      #self.HeadSensorToHeadSTL = slicer.util.getNode('HeadSensorToPegHeadS')
      self.HeadSensorToHeadSTL = slicer.util.getNode('HeadSensorToNewPegHe')
    else:
      self.HeadSensorToHeadSTL = slicer.util.getNode('HeadSensorToHeadSTL')
    try:
      self.ExtraTransform = slicer.util.getNode('Extra')
    except slicer.util.MRMLNodeNotFoundException:
      # Default to 6mm Z-axis offset (sensor is 6mm from tip of scope)
      self.ExtraTransform = self.createTransformNode(translationMm=[0,0,6],transformName='Extra')
      
    # Models
    logging.debug('Create models')

    try: 
      self.needleModel = slicer.util.getNode('NeedleModel')
    except slicer.util.MRMLNodeNotFoundException:
      self.needleModel = None
    if not self.needleModel:
      self.needleModel = slicer.modules.createmodels.logic().CreateNeedle(80, 1.0, 2.5, 0)
      self.needleModel.SetName('NeedleModel')

    # Build transform tree
    logging.debug('Set up transform tree')
    ## In our case, the transform tree is
    ## HeadSensorToHeadSTL > EmTrackerToHeadSenso > StylusSensorToEmTrac > StylusTipToStylusSen
    self.EmTrackerToHeadSensor.SetAndObserveTransformNodeID(self.HeadSensorToHeadSTL.GetID())
    self.StylusSensorToEmTracker.SetAndObserveTransformNodeID(self.EmTrackerToHeadSensor.GetID())
    #self.StylusTipToStylusSensor.SetAndObserveTransformNodeID(self.StylusSensorToEmTracker.GetID())
    self.NeedleTipToStylusSensor.SetAndObserveTransformNodeID(self.StylusSensorToEmTracker.GetID())
    # NOTE Choose one of the following two lines depending on which stylus/sensor type is appropriate
    usingScopeSensor = True # TODO: don't hard code this
    if usingScopeSensor:
      self.ExtraTransform.SetAndObserveTransformNodeID(self.NeedleTipToStylusSensor.GetID())
      self.needleModel.SetAndObserveTransformNodeID(self.ExtraTransform.GetID())
    else:
      # Using stylus sensor (plastic)
      self.needleModel.SetAndObserveTransformNodeID(self.StylusTipToStylusSensor.GetID())

    ## self.needleToReference.SetAndObserveTransformNodeID(self.referenceToRas.GetID())
    ## self.needleTipToNeedle.SetAndObserveTransformNodeID(self.needleToReference.GetID())
    ## self.needleModel.SetAndObserveTransformNodeID(self.needleTipToNeedle.GetID())

    # Set "Extra" as default leaf node for processing
    self.parameterNode.SetParameter('leafTransformNodeID', self.ExtraTransform.GetID()) 
    self.leafTransformNodeSelector.setCurrentNodeID(self.ExtraTransform.GetID())
    self.sceneLeafTransformNode = self.ExtraTransform
    
    return

  def recordingCommandCompleted(self, command, q):
    """ lifted from AirwayTrackerClass.py """
    statusText = "Recording "
    statusText = statusText + self.plusRemoteNode.GetRecordingStatusAsString(self.plusRemoteNode.GetRecordingStatus()) + " "
    statusText = statusText + self.plusRemoteNode.GetRecordingMessage() + " "
    logging.info(statusText)
    self.startStopRecordingButton.setToolTip(statusText)

  def createTransformNode(self, translationMm=[0,0,0],transformName='CreatedTransform'):
    """ Create a simple translation-only linear transform node from scratch """
    import numpy as np
    transformNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLinearTransformNode',transformName)
    transformMatrix = np.eye(4,dtype=float)
    transformMatrix[0:3,3] = translationMm
    transformNode.SetAndObserveMatrixTransformToParent(slicer.util.vtkMatrixFromArray(transformMatrix))
    return transformNode

  def disconnect(self):#TODO see connect
    logging.debug('ScoliUs.disconnect()')
    Guidelet.disconnect(self)
    # Disconnect buttons (moved from AirwayTrackerClass.preCleanup -> disconnect)
    self.startStopRecordingButton.disconnect('clicked(bool)', self.onStartStopRecordingClicked)

    # Remove observer to old parameter node
    if self.patientSLandmarks_Reference and self.patientSLandmarks_ReferenceObserver:
      self.patientSLandmarks_Reference.RemoveObserver(self.patientSLandmarks_ReferenceObserver)
      self.patientSLandmarks_ReferenceObserver = None

    #self.calibrationCollapsibleButton.disconnect('toggled(bool)', self.onPatientSetupPanelToggled)
    #self.exampleButton.disconnect('clicked(bool)', self.onExampleButtonClicked)


  def patientSetupPanel(self):
    logging.debug('patientSetupPanel')

    # Load from UI file
    moduleDir = os.path.dirname(__file__)
    uiFilePath = os.path.join(moduleDir, 'Resources', 'UI', 'TrackerUIMike.ui')
    loadedUIWidget = slicer.util.loadUI(uiFilePath)
    loadedUI = slicer.util.childWidgetVariables(loadedUIWidget)
    self.sliceletPanelLayout.addWidget(loadedUIWidget)
    
    self.startStopRecordingButton = loadedUI.StartStopRecordingButton
    self.startStopRecordingButton.setCheckable(True)
    self.startStopRecordingButton.setIcon(self.recordIcon)
    self.startStopRecordingButton.setToolTip("If clicked, start recording")

    self.saveUserInfoButton = loadedUI.SaveUserInfoButton
    self.experienceLevelComboBox = loadedUI.ExperienceLevelComboBox
    self.roleComboBox = loadedUI.RoleComboBox
    self.userNameLineEdit = loadedUI.UserNameLineEdit
    self.currentUserNameLabel = loadedUI.CurrentUserNameLabel
    self.currentExperienceLevelLabel = loadedUI.CurrentExperienceLevelLabel
    self.currentRoleLabel = loadedUI.CurrentRoleLabel

    self.runToReviewComboBox = loadedUI.RunToReviewComboBox
    self.expertRunToCompareComboBox = loadedUI.ExpertRunToCompareComboBox
    self.displaySelectedRunButton = loadedUI.DisplaySelectedRunButton

    ##
    """
    self.calibrationCollapsibleButton.setProperty('collapsedHeight', 20)
    self.calibrationCollapsibleButton.text = 'Calibration'
    self.sliceletPanelLayout.addWidget(self.calibrationCollapsibleButton)

    self.calibrationButtonLayout = qt.QFormLayout(self.calibrationCollapsibleButton)
    self.calibrationButtonLayout.setContentsMargins(12, 4, 4, 4)
    self.calibrationButtonLayout.setSpacing(4)

    self.exampleButton = qt.QPushButton("Example button")
    self.exampleButton.setCheckable(False)
    self.calibrationButtonLayout.addRow(self.exampleButton)
    """


  def onExampleButtonClicked(self, toggled):
    logging.debug('onExampleButtonClicked')


  def onPatientSetupPanelToggled(self, toggled):
    if toggled == False:
      return

    logging.debug('onPatientSetupPanelToggled: {0}'.format(toggled))

    #self.selectView(self.VIEW_ULTRASOUND_3D)

	
  def onUltrasoundPanelToggled(self, toggled):
    if not toggled:
      # deactivate placement mode
      interactionNode = slicer.app.applicationLogic().GetInteractionNode()
      interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
      return

    logging.debug('onUltrasoundPanelToggled: {0}'.format(toggled))

    self.selectView(self.VIEW_ULTRASOUND_3D)

    # The user may want to freeze the image (disconnect) to make contouring easier.
    # Disable automatic ultrasound image auto-fit when the user unfreezes (connect)
    # to avoid zooming out of the image.
    self.fitUltrasoundImageToViewOnConnect = not toggled



  def getCamera(self, viewName):
    """
    Get camera for the selected 3D view
    """
    camerasLogic = slicer.modules.cameras.logic()
    camera = camerasLogic.GetViewActiveCameraNode(slicer.util.getNode(viewName))
    return camera


  def getViewNode(self, viewName):
    """
    Get the view node for the selected 3D view
    """
    viewNode = slicer.util.getNode(viewName)
    return viewNode


  def updateNavigationView(self):
    self.selectView(self.navigationView)

    # Reset orientation marker
    if hasattr(slicer.vtkMRMLViewNode(),'SetOrientationMarkerType'): # orientation marker is not available in older Slicer versions
      v1=slicer.util.getNode('View1')
      v1.SetOrientationMarkerType(v1.OrientationMarkerTypeNone)
