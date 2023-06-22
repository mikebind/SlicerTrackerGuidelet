import os
from __main__ import vtk, qt, ctk, slicer

from SlicerGuideletBase import GuideletLoadable, GuideletLogic, GuideletTest, GuideletWidget
from SlicerGuideletBase import Guidelet
import logging
import time


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
        recordPrefix = self.guideletParent.parameterNode.GetParameter('RecordingFilenamePrefix')
        recordExt = self.guideletParent.parameterNode.GetParameter('RecordingFilenameExtension')
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


class ExampleGuideletLogic(GuideletLogic):
  """Uses GuideletLogic base class, available at:
  """ #TODO add path


  def __init__(self, parent = None):
    GuideletLogic.__init__(self, parent)

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
    defaultSceneSavePath = os.path.join(moduleDir, 'SavedScenes')
    moduleDirectoryPath = slicer.modules.exampleguidelet.path.replace('ExampleGuidelet.py','')
    settingList = {
                   'StyleSheet' : moduleDirectoryPath + 'Resources/StyleSheets/ExampleGuideletStyle.qss',
                   'LiveUltrasoundNodeName': 'Image_Reference',
                   'TestMode' : 'False',
                   'RecordingFilenamePrefix' : 'ExampleGuideletRec-',
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
    self.calibrationCollapsibleButton = None

    Guidelet.__init__(self, parent, logic, configurationName)
    logging.debug('ExampleGuideletGuidelet.__init__')

    self.logic.addValuesToDefaultConfiguration()

    moduleDirectoryPath = slicer.modules.exampleguidelet.path.replace('ExampleGuidelet.py', '')

    # Set up main frame.

    self.sliceletDockWidget.setObjectName('ExampleGuideletPanel')
    self.sliceletDockWidget.setWindowTitle('Example guidelet')
    self.mainWindow.setWindowTitle('ExampleGuidelet')
    self.mainWindow.windowIcon = qt.QIcon(moduleDirectoryPath + '/Resources/Icons/ExampleGuidelet.png')

    self.setupScene()

    self.navigationView = self.VIEW_ULTRASOUND_3D

    # Setting button open on startup.
    self.calibrationCollapsibleButton.setProperty('collapsed', False)


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
    logging.debug('preCleanup')


  def setupConnections(self):
    logging.debug('ScoliUs.setupConnections()')
    Guidelet.setupConnections(self)
    self.calibrationCollapsibleButton.connect('toggled(bool)', self.onPatientSetupPanelToggled)
    self.exampleButton.connect('clicked(bool)', self.onExampleButtonClicked)


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

    Guidelet.setupScene(self)
    # Not sure why 'EmTrackerToHeadSenso' didn't exist yet, trying processing events here
    slicer.app.processEvents() 

    # Transforms

    logging.debug('Create transforms')

    '''
    In this example we assume that there is a tracked needle in the system. The needle is not
    tracked at its tip, so we need a NeedleTipToNeedle transform to define where the needle tip is.
    In your application Needle may be called Stylus, or maybe you don't need such a tool at all.
    '''

    ## self.needleToReference = slicer.util.getNode('NeedleToReference')
    ## if not self.needleToReference:
    ##   self.needleToReference = slicer.vtkMRMLLinearTransformNode()
    ##   self.needleToReference.SetName('NeedleToReference')
    ##   slicer.mrmlScene.AddNode(self.needleToReference)

    ## self.needleTipToNeedle = slicer.util.getNode('NeedleTipToNeedle')
    ## if not self.needleTipToNeedle:
    ##   self.needleTipToNeedle = slicer.vtkMRMLLinearTransformNode()
    ##   self.needleTipToNeedle.SetName('NeedleTipToNeedle')
    ##   m = self.logic.readTransformFromSettings('NeedleTipToNeedle', self.configurationName)
    ##   if m:
    ##     self.needleTipToNeedle.SetMatrixTransformToParent(m)
    ##   slicer.mrmlScene.AddNode(self.needleTipToNeedle)

    # Use PegNeckHead?
    usingPegNeckHead = True
    slicer.app.processEvents()
    self.EmTrackerToHeadSensor = slicer.util.getNode('EmTrackerToHeadSenso')
    self.StylusSensorToEmTracker = slicer.util.getNode('StylusSensorToEmTrac')
    self.StylusTipToStylusSensor = slicer.util.getNode('StylusTipToStylusSen')
    self.NeedleTipToStylusSensor = slicer.util.getNode('NeedleTipToStylusSen')
    if usingPegNeckHead:
      #self.HeadSensorToHeadSTL = slicer.util.getNode('HeadSensorToPegHeadS')
      self.HeadSensorToHeadSTL = slicer.util.getNode('HeadSensorToNewPegHe')
    else:
      self.HeadSensorToHeadSTL = slicer.util.getNode('HeadSensorToHeadSTL')

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
    self.needleModel.SetAndObserveTransformNodeID(self.NeedleTipToStylusSensor.GetID())
    #self.needleModel.SetAndObserveTransformNodeID(self.StylusTipToStylusSensor.GetID())

    ## self.needleToReference.SetAndObserveTransformNodeID(self.referenceToRas.GetID())
    ## self.needleTipToNeedle.SetAndObserveTransformNodeID(self.needleToReference.GetID())
    ## self.needleModel.SetAndObserveTransformNodeID(self.needleTipToNeedle.GetID())


    # Hide slice view annotations (patient name, scale, color bar, etc.) as they
    # decrease reslicing performance by 20%-100%
    logging.debug('Hide slice view annotations')
    import DataProbe
    dataProbeUtil=DataProbe.DataProbeLib.DataProbeUtil()
    dataProbeParameterNode=dataProbeUtil.getParameterNode()
    dataProbeParameterNode.SetParameter('showSliceViewAnnotations', '0')


  def disconnect(self):#TODO see connect
    logging.debug('ScoliUs.disconnect()')
    Guidelet.disconnect(self)

    # Remove observer to old parameter node
    if self.patientSLandmarks_Reference and self.patientSLandmarks_ReferenceObserver:
      self.patientSLandmarks_Reference.RemoveObserver(self.patientSLandmarks_ReferenceObserver)
      self.patientSLandmarks_ReferenceObserver = None

    self.calibrationCollapsibleButton.disconnect('toggled(bool)', self.onPatientSetupPanelToggled)
    self.exampleButton.disconnect('clicked(bool)', self.onExampleButtonClicked)


  def patientSetupPanel(self):
    logging.debug('patientSetupPanel')

    self.calibrationCollapsibleButton.setProperty('collapsedHeight', 20)
    self.calibrationCollapsibleButton.text = 'Calibration'
    self.sliceletPanelLayout.addWidget(self.calibrationCollapsibleButton)

    self.calibrationButtonLayout = qt.QFormLayout(self.calibrationCollapsibleButton)
    self.calibrationButtonLayout.setContentsMargins(12, 4, 4, 4)
    self.calibrationButtonLayout.setSpacing(4)

    self.exampleButton = qt.QPushButton("Example button")
    self.exampleButton.setCheckable(False)
    self.calibrationButtonLayout.addRow(self.exampleButton)


  def onExampleButtonClicked(self, toggled):
    logging.debug('onExampleButtonClicked')


  def onPatientSetupPanelToggled(self, toggled):
    if toggled == False:
      return

    logging.debug('onPatientSetupPanelToggled: {0}'.format(toggled))

    self.selectView(self.VIEW_ULTRASOUND_3D)

	
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
