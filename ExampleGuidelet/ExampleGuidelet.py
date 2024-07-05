import os
import pathlib

# from __main__ import vtk, qt, ctk, slicer
import slicer, vtk, qt, ctk
from slicer import util

import typing
from typing import Optional, Union, Iterable, MutableMapping, Tuple, List, Type, Dict
from slicer import (
    vtkMRMLModelNode,
    vtkMRMLTransformNode,
    vtkMRMLLinearTransformNode,
)

from SlicerGuideletBase import (
    GuideletLoadable,
    GuideletLogic,
    GuideletTest,
    GuideletWidget,
)
from SlicerGuideletBase import Guidelet
import logging
import time
import numpy as np
import Lib.HelperClasses  # allows access to methods outside of the classes
from Lib.HelperClasses import Session, Recording, ScopeRun

# Lib.HelperClasses.loadOnlyScopeRunsFromSessionFile()


class ExampleGuidelet(GuideletLoadable):
    """Uses GuideletLoadable class, available at:"""

    def __init__(self, parent):
        GuideletLoadable.__init__(self, parent)
        self.parent.title = "AirwayScopeTrainerGuidelet"
        self.parent.categories = ["MikeTools"]
        self.parent.dependencies = []
        self.parent.contributors = ["Mike Bindschadler"]
        self.parent.helpText = """ Simple Guidelet based on template and used for a trainer for airway flexible endoscopy """
        self.parent.acknowledgementText = (
            """ Supported by Seattle Children's Hospital """
        )


class ExampleGuideletWidget(GuideletWidget):
    """Uses GuideletWidget base class, from SlicerGuideletBase/GuideletLoadable.py
    This class manages the guidelet launcher widgets, but not the main
    guidelet gui widgets (I think). The main guidelet gui is managed in the
    ExampleGuideletGuidelet class.
    """

    def __init__(self, parent=None):
        GuideletWidget.__init__(self, parent)
        self.startStopRecordingButton = None
        self.captureDeviceName = ""

    def setup(self):
        GuideletWidget.setup(self)
        fileDir = os.path.dirname(__file__)
        iconPathRecord = os.path.join(fileDir, "Resources", "Icons", "icon_Record.png")
        iconPathStop = os.path.join(fileDir, "Resources", "Icons", "icon_Stop.png")

        if os.path.isfile(iconPathRecord):
            self.recordIcon = qt.QIcon(iconPathRecord)
        if os.path.isfile(iconPathStop):
            self.stopIcon = qt.QIcon(iconPathStop)

    def addLauncherWidgets(self):
        GuideletWidget.addLauncherWidgets(self)

    def onConfigurationChanged(self, selectedConfigurationName):
        GuideletWidget.onConfigurationChanged(self, selectedConfigurationName)
        # settings = slicer.app.userSettings()

    def addBreachWarningLightPreferences(self):
        pass

    def onBreachWarningLightChanged(self, state):
        pass

    def createGuideletInstance(self):
        """Override of abstract method in GuideletLoadable, called from
        onLaunchGuideletButtonClicked()
        """
        return ExampleGuideletGuidelet(
            None, self.guideletLogic, self.selectedConfigurationName
        )

    def createGuideletLogic(self):
        """Override of abstract method in GuideletLoadable, called from
        GuideletWidget.__init__()
        """
        return ExampleGuideletLogic()


HEAD_SENSOR_TRANSFORM_POSITION_IN_HIERARCHY = 1
SCOPE_SENSOR_TRANSFORM_POSITION_IN_HIERARCHY = 2
DEFAULT_LEAF_TRANSFORM_NODE_NAME = "Extra"
moduleDir = os.path.dirname(__file__)
segDir = os.path.join(moduleDir, "Resources", "Segmentations")
segDir2024F = os.path.join(moduleDir, "Resources", "Segmentations", "BootCamp2024Final")

PEGNECK_AIRWAYZONE_SEGMENTATION = os.path.join(
    segDir, "airwayZoneSegmentation.seg.nrrd"
)

RIGIDNECK_AIRWAYZONE_SEGMENTATION = os.path.join(
    segDir, "RigidNeckAirwaySegmentation.seg.nrrd"
)
RIGIDNECK_STL = os.path.join(segDir, "SolidOuter_Cropped.stl")

SUPINE_AIRWAYZONE_SEGMENTATION = os.path.join(segDir, "SupineScanSegmentation.seg.nrrd")
SUPINE_STL = os.path.join(segDir, "SupineSinusModel.vtk")
SUPINE_IMAGE = os.path.join(segDir, "SupineCroppedImage.nrrd")
JULY9_AIRWAYZONE_SEGMENTATION = os.path.join(
    segDir, "July9ScanAirwayZoneSegmentation.seg.nrrd"
)
JULY9_OUTERMODEL_STL = os.path.join(
    segDir,
    "July9ScanAirwayZoneSegmentation_OuterSupineSinusModel.stl",
)
JULY9_IMAGE = os.path.join(segDir, "July9_AxBone11_cropped1mm.nrrd")

AIRWAY_PRACTICE_2024_AIRWAYZONE_SEGMENTATION = os.path.join(
    segDir, "SoundsSegmentationHardToJ9.seg.nrrd"
)
AIRWAY_PRACTICE_2024_IMAGE = os.path.join(
    segDir,
    "AIRWAY TESTING_Silicone Nose_PracticeModel_Scan_1mm.nrrd",
)
AIRWAY_PRACTICE_2024_OUTERMODEL_STL = os.path.join(
    segDir,
    "PrintedPlasticSimpDecim_2024.stl",
)

Final2024_OUTERMODEL_STL = os.path.join(segDir2024F, "PrintedPlasticSolidApprox.stl")
Final2024_AIRWAYZONE_SEGMENTATION = os.path.join(
    segDir2024F, "airwayZoneOnly_LowRes2mm.seg.nrrd"
)
Final2024_IMAGE = os.path.join(segDir2024F, "Final2024BootCamp_1mm.nrrd")
FOR_VIEWPOINT_TRANSFORM_2024_SCOPE1 = os.path.join(segDir2024F, "ForViewpoint.h5")

COUGH_ZONE_MODEL_STL = os.path.join(segDir2024F, "CoughZoneTrimmed.stl")
COUGH_ZONE_COLOR = (0.945098, 0.839216, 0.568627)
GAG_ZONE_MODEL_STL = os.path.join(segDir2024F, "GagZone.stl")
GAG_ZONE_COLOR = (0.694118, 0.478431, 0.396078)
OUCH2_ZONE_MODEL_STL = os.path.join(segDir2024F, "OuchZone2.stl")
OUCH2_ZONE_COLOR = (0.501961, 0.682353, 0.501961)
SEPTUM_ZONE_MODEL_STL = os.path.join(segDir2024F, "SeptumZoneTrimmed.stl")
SEPTUM_ZONE_COLOR = (0.5647, 0.9333, 0.5647)
# TEST_ZONE_MODEL_STL = os.path.join(segDir, "testSoundZone.stl")
# Sound Paths
soundDir = os.path.join(moduleDir, "Resources", "Sounds")
COUGH_SOUND_PATH = pathlib.Path(soundDir, "cough1.wav")
GAG_SOUND_PATH = pathlib.Path(soundDir, "gag1.wav")  # or gag2.wav
SEPTUM_SOUND_PATH = pathlib.Path(soundDir, "OwMySeptum.wav")
OUCH2_SOUND_PATH = pathlib.Path(soundDir, "Ow.wav")
MOUTH_SOUND_PATH = pathlib.Path(soundDir, "Mouth.wav")
RIGHT_NOSTRIL_SOUND_PATH = pathlib.Path(soundDir, "RightNostril.wav")
TEST_SOUND_PATH = pathlib.Path(soundDir, "testZoneSound.wav")


# MARK: ExampleGuideletLogic
class ExampleGuideletLogic(GuideletLogic):
    """Uses GuideletLogic base class, available at:"""  # TODO add path

    def __init__(self, parent=None):
        GuideletLogic.__init__(self, parent)

    def centerSlicesOnTransformedPoint(self, leafNode):
        # Jump all slices to the origin point of a given transform node (taking into account all parent transforms)
        m = vtk.vtkMatrix4x4()
        leafNode.GetMatrixTransformBetweenNodes(leafNode, None, m)
        position_RAS = [
            m.GetElement(0, 3),
            m.GetElement(1, 3),
            m.GetElement(2, 3),
        ]  # same as multiplying matrix by [0,0,0,1]
        slicer.vtkMRMLSliceNode.JumpAllSlices(
            slicer.mrmlScene, *position_RAS, slicer.vtkMRMLSliceNode.CenteredJumpSlice
        )

    def startLiveUpdate(self, leafTransformNode):
        callback = lambda unused1, unused2: self.centerSlicesOnTransformedPoint(
            leafTransformNode
        )  # unused inputs are caller and event arguments to event callbacks
        self.liveUpdateCallbackID = leafTransformNode.AddObserver(
            slicer.vtkMRMLTransformNode.TransformModifiedEvent, callback
        )
        self.liveUpdateLeafNode = leafTransformNode
        # callback ID and leaf node are stored so that stopLiveUpdate can run without any inputs and still properly stop the live update
        return self.liveUpdateCallbackID

    def stopLiveUpdate(self):
        self.liveUpdateLeafNode.RemoveObserver(self.liveUpdateCallbackID)
        # TODO: make this robust to errors like calling startLiveUpdate twice before calling stopLiveUpdate

    def displayScopeRun(self, scopeRunToDisplay):
        logging.debug("ExampleGuideletLogic.displayScopeRun()")
        if scopeRunToDisplay.coneModel is None:
            scopeRunToDisplay.createModelNodes()
        scopeRunToDisplay.showModelNodes()

    def hideScopeRun(self, scopeRunToDisplay):
        logging.debug("ExampleGuideletLogic.hideScopeRun()")
        scopeRunToDisplay.hideModelNodes()

    def mhaTesting(self, mhaFile=None):
        if mhaFile is None:
            # mhaFile = r'C:\Users\mikeb\Downloads\ExampleGuideletRec-20230601-113809.mhd'
            mhaFile = r"C:/Users/mikeb/Downloads/MayaTanyaRunsData/MayaTanyaRunsData/ExampleGuideletRec-20230602-083158.mhd"
            # mhaFile = r"C:/Users/mikeb/Downloads/MayaTanyaRunsData/MayaTanyaRunsData/ExampleGuideletRec-20230602-073923.mhd"
            # mhaFile = r'C:\Users\mikeb\Downloads\ExampleGuideletRec-20230602-080018.mhd'
            # mhaFile = r'C:/Users/mikeb/Downloads/Recording.igs20230524_103059.mha'
        (
            timeStamps,
            headSensorTransforms,
            scopeSensorTransforms,
        ) = self.import_tracker_recording(mhaFile)
        # Set up the hierarchy order
        transformsList = self.gatherTestingTransforms()
        transformsList[1] = headSensorTransforms
        transformsList[2] = scopeSensorTransforms
        positions = self.positions_from_transform_hierarchy(transformsList)
        curveNode = self.markupsCurveFromPositions(positions)

        return (
            timeStamps,
            curveNode,
            positions,
            headSensorTransforms,
            scopeSensorTransforms,
        )

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

        headSToStlNode = getNode("HeadSensorToHeadSTL")
        TrToHeadS = getNode("EmTrackerToHeadSenso")
        ScopeSToTr = getNode("StylusSensorToEmTrac")
        ScopeTipToScopeS = getNode("NeedleTipToStylusSen")
        ExtraTipAdjustment = getNode("extra")
        TNodeList = [
            headSToStlNode,
            TrToHeadS,
            ScopeSToTr,
            ScopeTipToScopeS,
            ExtraTipAdjustment,
        ]
        transformList = [
            slicer.util.arrayFromTransformMatrix(TNode) for TNode in TNodeList
        ]
        return transformList

    def markupsCurveFromPositions(self, positions):
        """Create markupsCurveNode from Nx3 numpy array"""
        pathName = slicer.mrmlScene.GenerateUniqueName("Path")
        curveNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode", pathName
        )
        slicer.util.updateMarkupsControlPointsFromArray(curveNode, positions)
        return curveNode

    def addValuesToDefaultConfiguration(self):
        GuideletLogic.addValuesToDefaultConfiguration(self)
        moduleDir = os.path.dirname(slicer.modules.exampleguidelet.path)
        defaultUserSessionsSavePath = os.path.join(
            moduleDir, "UserSessionResults"
        )  # TODO: Create folder if it doesn't exist
        defaultSceneSavePath = os.path.join(moduleDir, "SavedScenes")
        moduleDirectoryPath = slicer.modules.exampleguidelet.path.replace(
            "ExampleGuidelet.py", ""
        )
        settingList = {
            "StyleSheet": moduleDirectoryPath
            + "Resources/StyleSheets/ExampleGuideletStyle.qss",
            "LiveUltrasoundNodeName": "Image_Reference",
            "TestMode": "False",
            "RecordingFilenamePrefix": "AirwayTrackerRec-",
            "UserSessionResultsDirectory": defaultUserSessionsSavePath,  # folder to put session files in
            "SavedScenesDirectory": defaultSceneSavePath,  # overwrites the default setting param of base
            "testParameter": "DoesThisShowUp?",
            "defaultSoundDistanceThresholdMm": "2.0",
            "septumZoneSoundDistThreshMm": "2.0",
            "ouch2ZoneSoundDistThreshMm": "1.0",
            "gagZoneSoundDistThreshMm": "3.0",
            "coughZoneSoundDistThreshMm": "2.0",
        }
        self.updateSettings(settingList, "Default")

    def setupBreachSound(
        self,
        soundFilePath: pathlib.Path,
        tipTransform: vtkMRMLTransformNode,
        watchedModel: vtkMRMLModelNode,
        distanceThresholdMm: float = 5.0,
        outputBreachWarningNode=None,
        showLinkingLine=False,
        breachNodeName=None,
        linkingLineName="d",
    ):
        """Set up a sound to play when a model is approached too closely.
        The sound file must be .wav.  Negative distance thresholds would
        be inside the model (use zero or positive).  The mechanism used
        is a vtkMRMLBreachWarningNode.
        """
        sound = qt.QSoundEffect()
        sourceUrl = qt.QUrl.fromLocalFile(soundFilePath.as_posix())
        sound.setSource(sourceUrl)
        # Can check if loading went OK by checking sound.status
        # Should probably warn here if it didn't load properly
        if outputBreachWarningNode is None:
            outputBreachWarningNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLBreachWarningNode"
            )
        outputBreachWarningNode.SetAndObserveToolTransformNodeId(tipTransform.GetID())
        outputBreachWarningNode.SetAndObserveWatchedModelNodeID(watchedModel.GetID())
        outputBreachWarningNode.SetWarningDistanceMM(distanceThresholdMm)
        outputBreachWarningNode.SetPlayWarningSound(
            False
        )  # wav file should be played rather than beep

        # Regardless of showLinkingLine value, initially show it to create
        # line node, then hide if it is not supposed to be shown
        slicer.modules.breachwarning.logic().SetLineToClosestPointVisibility(
            True, outputBreachWarningNode
        )
        if not showLinkingLine:
            # Hide the linking line
            slicer.modules.breachwarning.logic().SetLineToClosestPointVisibility(
                False, outputBreachWarningNode
            )
        # Set line name
        outputBreachWarningNode.GetLineToClosestPointNode().SetName(linkingLineName)
        # Set up observer
        callbackFcn = lambda unused1, unused2: self.zoneModelModified(
            outputBreachWarningNode, sound
        )
        # The lambda is needed because the callback is going to get two extra inputs which
        # aren't needed or used
        observerTag = watchedModel.GetDisplayNode().AddObserver(
            vtk.vtkCommand.ModifiedEvent, callbackFcn
        )
        # Apply name
        if breachNodeName is not None:
            outputBreachWarningNode.SetName(breachNodeName)
        return outputBreachWarningNode, (watchedModel, observerTag, callbackFcn)

    def zoneModelModified(self, assocBreachWarningNode, soundEffect):
        """Trigger to possibly play sound effect, only if the associated
        breach warning node reports less than threshold distance, and only
        if the soundEffect is not already playing.
        This is the callback assigned to modification of the watched model
        (breach warning triggers color change, triggering Modified event,
        which launches this callback)
        """
        logging.debug(
            "zoneModelModified() triggered with breach node watching the display node of %s",
            assocBreachWarningNode.GetWatchedModelNode().GetID(),
        )
        if (
            assocBreachWarningNode.IsToolTipInsideModel()
            and not soundEffect.isPlaying()
        ):
            soundEffect.play()


# MARK: ExampleGuideletTest
class ExampleGuideletTest(GuideletTest):
    """This is the test case for your scripted module."""

    def runTest(self):
        """Run as few or as many tests as needed here."""
        GuideletTest.runTest(self)
        # self.test_ExampleGuidelet1() #add applet specific tests here


# MARK: ExampleGuideletGuidelet
class ExampleGuideletGuidelet(Guidelet):
    def __init__(self, parent, logic, configurationName="Default"):
        # self.calibrationCollapsibleButton = None
        try:
            slicer.modules.plusremote
        except:
            raise Exception(
                "Error: Could not find Plus Remote module. Please install the SlicerOpenIGTLink extension"
            )

        self.plusRemoteLogic = slicer.modules.plusremote.logic()
        self.plusRemoteNode = None
        # Set up icon paths
        fileDir = os.path.dirname(__file__)
        iconPathRecord = os.path.join(fileDir, "Resources", "Icons", "icon_Record.png")
        iconPathStop = os.path.join(fileDir, "Resources", "Icons", "icon_Stop.png")

        if os.path.isfile(iconPathRecord):
            self.recordIcon = qt.QIcon(iconPathRecord)
        else:
            logging.warning(f"Icon not found at {iconPathRecord}!")
        if os.path.isfile(iconPathStop):
            self.stopIcon = qt.QIcon(iconPathStop)

        # Init guidelet
        Guidelet.__init__(self, parent, logic, configurationName)
        # self.parameterNode is created in Guidelet.__init__()
        self.zoneModelObserversList = []
        self._updatingGuideletGUIFromParameterNode = False
        self.updateParameterNodeFromGuideletGUI()  # force initial update from loaded GUI values (could also set up parameter node
        # ahead of time, but if we don't do either we end up trying to update the GUI from empty parameter node fields)

        logging.debug("ExampleGuideletGuidelet.__init__")

        # TODO: understand what this next line really does
        self.logic.addValuesToDefaultConfiguration()

        moduleDirectoryPath = slicer.modules.exampleguidelet.path.replace(
            "ExampleGuidelet.py", ""
        )

        # Set up main frame.

        self.sliceletDockWidget.setObjectName("ExampleGuideletPanel")
        self.sliceletDockWidget.setWindowTitle("Airway Tracker")
        self.mainWindow.setWindowTitle("Airway Tracker")
        self.mainWindow.windowIcon = qt.QIcon(
            moduleDirectoryPath + "/Resources/Icons/ExampleGuidelet.png"
        )
        # Load image, segmentation, models
        self.setupScene()

        self.navigationView = self.VIEW_3D

        # Setting button open on startup.
        # self.calibrationCollapsibleButton.setProperty('collapsed', False)
        self.scopeRunsDisplayed = []  # initalize, no runs showing right now

        # Set up sounds to be able to play as warnings
        self.setupSounds()

    def updateParameterNodeFromGuideletGUI(self, caller=None, event=None):
        """Update parameter node values from current GUI information"""
        self.parameterNode.SetParameter(
            "userNameLineEditString", self.userNameLineEdit.text
        )
        self.parameterNode.SetParameter(
            "experienceLevelComboBoxCurrentIndex",
            str(self.experienceLevelComboBox.currentIndex),
        )
        self.parameterNode.SetParameter(
            "experienceLevelComboBoxCurrentString",
            self.experienceLevelComboBox.currentText,
        )
        self.parameterNode.SetParameter(
            "roleComboBoxCurrentIndex", str(self.roleComboBox.currentIndex)
        )
        self.parameterNode.SetParameter(
            "roleComboBoxCurrentText", self.roleComboBox.currentText
        )
        self.parameterNode.SetNodeReferenceID(
            "airwayZoneSegmentationNode",
            self.airwayZoneSegmentationNodeSelector.currentNodeID,
        )
        self.parameterNode.SetNodeReferenceID(
            "sceneLeafTransformNode", self.leafTransformNodeSelector.currentNodeID
        )

    def updateGuideletGUIFromParameterNode(self, caller=None, event=None):
        """Update Guidelet GUI elements from parameter values"""
        if self.parameterNode is None or self._updatingGuideletGUIFromParameterNode:
            return
        self._updatingGuideletGUIFromParameterNode = (
            True  # prevent infinite update loops
        )
        # User name
        self.userNameLineEdit.text = self.parameterNode.GetParameter(
            "userNameLineEditString"
        )
        # Experience level
        self.experienceLevelComboBox.setCurrentIndex(
            int(self.parameterNode.GetParameter("experienceLevelComboBoxCurrentIndex"))
        )
        # current text is updated automatically with the index update ^^
        # Role
        self.roleComboBox.setCurrentIndex(
            int(self.parameterNode.GetParameter("roleComboBoxCurrentIndex"))
        )
        # AirwayZone Segmentation
        self.airwayZoneSegmentationNodeSelector.setCurrentNodeID(
            self.parameterNode.GetNodeReferenceID("airwayZoneSegmentationNode")
        )
        self.leafTransformNodeSelector.setCurrentNodeID(
            self.parameterNode.GetNodeReferenceID("sceneLeafTransformNode")
        )
        # List of runs
        # List of expert runs
        # NOTE: Parameter node settings related to the Advanced panel are handled in Guidelet.py
        self._updatingGuideletGUIFromParameterNode = False

    def createFeaturePanels(self):
        # Create GUI panels.

        self.calibrationCollapsibleButton = ctk.ctkCollapsibleButton()
        self.patientSetupPanel()

        featurePanelList = Guidelet.createFeaturePanels(self)

        featurePanelList[len(featurePanelList) :] = [self.calibrationCollapsibleButton]

        return featurePanelList

    def removeZoneModelObservers(self):
        """Remove the observers watching the various breach warning watched models,
        if present
        """
        for model, observerTag, callbackFcn in self.zoneModelObserversList:
            model.RemoveObserver(observerTag)
        self.zoneModelObserversList = []

    def __del__(self):  # common
        self.preCleanup()

    # Clean up when guidelet is closed
    def preCleanup(self):  # common
        Guidelet.preCleanup(self)
        self.disconnect()
        self.removeZoneModelObservers()

        logging.debug("preCleanup")

    def createPlusConnector(self):
        connectorNode = slicer.mrmlScene.GetFirstNodeByName("PlusConnector")
        if not connectorNode:
            connectorNode = slicer.vtkMRMLIGTLConnectorNode()
            slicer.mrmlScene.AddNode(connectorNode)
            connectorNode.SetName("PlusConnector")
            hostNamePort = self.parameterNode.GetParameter(
                "PlusServerHostNamePort"
            )  # example: "localhost:18944"
            [hostName, port] = hostNamePort.split(":")
            connectorNode.SetTypeClient(hostName, int(port))
            logging.debug("PlusConnector created")
        return connectorNode

    def onConnectorNodeConnected_Ex(self):
        # self.freezeUltrasoundButton.setText('Freeze')
        self.startStopRecordingButton.setEnabled(True)

    def onConnectorNodeDisconnected_Ex(self):
        # self.freezeUltrasoundButton.setText('Un-freeze')
        if (
            self.parameterNode.GetParameter(
                "RecordingEnabledWhenConnectorNodeDisconnected"
            )
            == "False"
        ):
            self.startStopRecordingButton.setEnabled(False)

    def setupConnections(self):
        logging.debug("ExampleGuideletGuidelet.setupConnections()")
        Guidelet.setupConnections(self)
        self.startStopRecordingButton.connect(
            "clicked(bool)", self.onStartStopRecordingClicked
        )
        self.userNameLineEdit.connect(
            "editingFinished()", self.updateParameterNodeFromGuideletGUI
        )
        self.roleComboBox.connect(
            "currentIndexChanged(int)", self.updateParameterNodeFromGuideletGUI
        )
        self.experienceLevelComboBox.connect(
            "currentIndexChanged(int)", self.updateParameterNodeFromGuideletGUI
        )
        self.saveUserInfoButton.connect("clicked(bool)", self.saveUserInfoButtonClicked)
        self.displaySelectedRunButton.connect(
            "clicked(bool)", self.onDisplaySelectedRunClicked
        )
        self.liveUpdateCheckBox.connect(
            "toggled(bool)", self.onLiveUpdateCheckBoxToggled
        )

        # self.calibrationCollapsibleButton.connect('toggled(bool)', self.onPatientSetupPanelToggled)
        # self.exampleButton.connect('clicked(bool)', self.onExampleButtonClicked)
        # TODO: Ensure disconnect() has all matching disconnections

    def setupSounds(self):
        """Set up all sounds which should be triggered by touching airway
        walls in certain places.  Create QSoundEffect resources and
        link them to models and distance thresholds.
        """
        pn = self.parameterNode
        leafTransformNode = pn.GetNodeReference("sceneLeafTransformNode")

        # Cough
        coughZoneModel = pn.GetNodeReference("coughZoneModel")
        coughSoundPath = pathlib.Path(pn.GetParameter("coughSoundPath"))
        coughDistThresh = float(pn.GetParameter("coughZoneSoundDistThreshMm"))
        coughBreachNode, coughObsInfo = self.logic.setupBreachSound(
            coughSoundPath,
            leafTransformNode,
            coughZoneModel,
            coughDistThresh,
            breachNodeName="coughBreach",
            linkingLineName="c",
        )
        self.zoneModelObserversList.append(coughObsInfo)
        pn.SetNodeReferenceID("coughBreachNode", coughBreachNode.GetID())
        # Gag
        gagZoneModel = pn.GetNodeReference("gagZoneModel")
        gagSoundPath = pathlib.Path(pn.GetParameter("gagSoundPath"))
        gagDistThresh = float(pn.GetParameter("gagZoneSoundDistThreshMm"))
        gagBreachNode, gagObsInfo = self.logic.setupBreachSound(
            gagSoundPath,
            leafTransformNode,
            gagZoneModel,
            gagDistThresh,
            breachNodeName="gagBreach",
            linkingLineName="g",
        )
        pn.SetNodeReferenceID("gagBreachNode", gagBreachNode.GetID())
        self.zoneModelObserversList.append(gagObsInfo)
        # Septum
        septumZoneModel = pn.GetNodeReference("septumZoneModel")
        septumSoundPath = pathlib.Path(pn.GetParameter("septumSoundPath"))
        septumDistThresh = float(pn.GetParameter("septumZoneSoundDistThreshMm"))
        septumBreachNode, septumObsInfo = self.logic.setupBreachSound(
            septumSoundPath,
            leafTransformNode,
            septumZoneModel,
            septumDistThresh,
            breachNodeName="septumBreach",
            linkingLineName="s",
        )
        pn.SetNodeReferenceID("septumBreachNode", septumBreachNode.GetID())
        self.zoneModelObserversList.append(septumObsInfo)
        # OuchZone2 (straight back poke with tip)
        ouch2ZoneModel = pn.GetNodeReference("ouch2ZoneModel")
        ouch2SoundPath = pathlib.Path(pn.GetParameter("ouch2SoundPath"))
        ouch2DistThresh = float(pn.GetParameter("ouch2ZoneSoundDistThreshMm"))
        ouch2BreachNode, ouch2ObsInfo = self.logic.setupBreachSound(
            ouch2SoundPath,
            leafTransformNode,
            ouch2ZoneModel,
            ouch2DistThresh,
            breachNodeName="ouch2Breach",
            linkingLineName="o",
        )
        pn.SetNodeReferenceID("ouch2BreachNode", ouch2BreachNode.GetID())
        self.zoneModelObserversList.append(ouch2ObsInfo)
        # TestZone
        # testZoneModel = pn.GetNodeReference("testZoneModel")
        # testZoneSoundPath = pathlib.Path(pn.GetParameter("testZoneSoundPath"))
        # testBreachNode, testObsInfo = self.logic.setupBreachSound(
        #     testZoneSoundPath, leafTransformNode, testZoneModel, distThreshMm
        # )
        # pn.SetNodeReferenceID("testBreachNode", testBreachNode.GetID())
        # self.zoneModelObserversList.append(testObsInfo)

    def onLiveUpdateCheckBoxToggled(self, tf: bool):
        """Toggle whether live updating is occuring"""
        if self.liveUpdateCheckBox.checked:
            self.selectView(self.VIEW_4UP)
            self.showSliceIntersctions(True)
            leafTransformNode = self.parameterNode.GetNodeReference(
                "sceneLeafTransformNode"
            )
            self.liveUpdateObserverId = self.logic.startLiveUpdate(leafTransformNode)
        else:
            self.logic.stopLiveUpdate()

    def showSliceIntersctions(self, tf: bool):
        if tf:
            visiblity = 1
        else:
            visiblity = 0
        sliceDisplayNodes = slicer.util.getNodesByClass("vtkMRMLSliceDisplayNode")
        for sliceDisplayNode in sliceDisplayNodes:
            sliceDisplayNode.SetIntersectingSlicesVisibility(visiblity)
        # Workaround to force visual update (see https://github.com/Slicer/Slicer/issues/6338)
        sliceNodes = slicer.util.getNodesByClass("vtkMRMLSliceNode")
        for sliceNode in sliceNodes:
            sliceNode.Modified()

    def onDisplaySelectedRunClicked(self):
        """Display the currently selected run in the 3D view"""
        logging.debug("ExampleGuideletGuidelet.onDisplaySelectedRunClicked()")
        key = self.runToReviewComboBox.currentText  # get from parameter node instead?
        scopeRunToDisplay = self.scopeRunDict[key]
        # Hide other displayed scope runs
        for S in self.scopeRunsDisplayed:
            self.logic.hideScopeRun(S)
        self.scopeRunsDisplayed = []  # TODO make display/hiding much more flexible!!
        self.logic.displayScopeRun(scopeRunToDisplay)
        # Track that this run is showing
        self.scopeRunsDisplayed.append(scopeRunToDisplay)

    def saveUserInfoButtonClicked(self, tf: bool):
        """Update the current user text section and save a session text file"""
        # User
        userText = self.parameterNode.GetParameter("userNameLineEditString")
        if not userText:
            userText = "*no current user saved*"
        self.parameterNode.SetParameter("CurrentUserText", userText)
        self.currentUserNameLabel.text = userText
        # Experience level
        experienceText = self.parameterNode.GetParameter(
            "experienceLevelComboBoxCurrentString"
        )
        if not experienceText:
            experienceText = "*not set*"
        self.parameterNode.SetParameter(
            "CurrentUserExperienceLevelText", experienceText
        )
        self.currentExperienceLevelLabel.text = experienceText
        # Role
        roleText = self.parameterNode.GetParameter("roleComboBoxCurrentText")
        if not roleText:
            roleText = "*not set*"
        self.parameterNode.SetParameter("CurrentUserRoleText", roleText)
        self.currentRoleLabel.text = roleText
        # Create Session File to hold results
        sessionDirectory = self.parameterNode.GetParameter(
            "UserSessionResultsDirectory"
        )

        userName = self.parameterNode.GetParameter("CurrentUserText")
        userDict = {
            "userName": userName,
            "experienceLevel": experienceText,
            "role": roleText,
        }
        self.currentSession = Session(userDict)
        self.currentSession.saveToFile(sessionDirectory)
        self.parameterNode.SetParameter(
            "CurrentSessionFilePath", self.currentSession.savedFilePathName
        )
        # Clear out RunsData for previous session
        self.updateRunsToReview()
        for S in self.scopeRunsDisplayed:
            self.logic.hideScopeRun(S)
        # delim = '|' # list delimiter for parameter node lists #TODO: store in to parameter node
        # self.parameterNode.SetParameter('RunsData', delim.join(listOfRuns))

    def onStartStopRecordingClicked(self):
        self.captureDeviceName = self.parameterNode.GetParameter(
            "PLUSCaptureDeviceName"
        )
        if self.startStopRecordingButton.isChecked():
            self.startStopRecordingButton.setText("  Stop Recording")
            self.startStopRecordingButton.setIcon(self.stopIcon)
            self.startStopRecordingButton.setToolTip("Recording is being started...")
            if self.captureDeviceName != "":
                # Important to save as .mhd because that does not require lengthy finalization (merging into a single file)
                recordPrefix = self.parameterNode.GetParameter(
                    "RecordingFilenamePrefix"
                )
                recordExt = self.parameterNode.GetParameter(
                    "RecordingFilenameExtension"
                )
                userName = self.parameterNode.GetParameter("CurrentUserText")
                userExp = self.parameterNode.GetParameter(
                    "CurrentUserExperienceLevelText"
                )
                userRole = self.parameterNode.GetParameter("CurrentUserRoleText")
                timeStamp = time.strftime(r"%Y-%m-%d-%H%M%S")
                self.recordingFileName = f"{recordPrefix}{userName}-{userExp}-{userRole}-{timeStamp}{recordExt}".replace(
                    " ", "_"
                )  # replace spaces with underscores
                # self.recordingFileName =  recordPrefix + time.strftime("%Y%m%d-%H%M%S") + recordExt

                logging.info(
                    "Starting recording to: {0}".format(self.recordingFileName)
                )

                self.plusRemoteNode.SetCurrentCaptureID(self.captureDeviceName)
                self.plusRemoteNode.SetRecordingFilename(self.recordingFileName)
                self.plusRemoteLogic.StartRecording(self.plusRemoteNode)

        else:
            self.startStopRecordingButton.setText("  Start Recording")
            self.startStopRecordingButton.setIcon(self.recordIcon)
            self.startStopRecordingButton.setToolTip("Recording is being stopped...")
            if self.captureDeviceName != "":
                logging.info("Stopping recording")
                self.plusRemoteNode.SetCurrentCaptureID(self.captureDeviceName)
                self.plusRemoteLogic.StopRecording(self.plusRemoteNode)
                # Add the new recording to the current session
                recordingsDirectory = self.parameterNode.GetParameter(
                    "PlusAppDataDirectory"
                )
                recordingFileFullPath = os.path.normpath(
                    os.path.join(recordingsDirectory, self.recordingFileName)
                ).replace("\\", "/")
                newRecording = Recording(self.currentSession, recordingFileFullPath)
                self.currentSession.addRecording(newRecording)
                leafTransformNode = self.parameterNode.GetNodeReference(
                    "sceneLeafTransformNode"
                )
                airwayZoneSegmentationNode = self.parameterNode.GetNodeReference(
                    "airwayZoneSegmentationNode"
                )
                # NOTE: running into a bug here where the recording file is not yet available when processing
                # tries to access it. Need to delay if file is not yet available
                max_attempts = 100
                attempt_count = 1
                while not os.path.exists(recordingFileFullPath):
                    time.sleep(0.1)
                    attempt_count += 1
                    if attempt_count > max_attempts:
                        break
                if not os.path.exists(recordingFileFullPath):
                    raise (
                        Exception(
                            f"Tried and failed {attempt_count} attempts to access {recordingFileFullPath}!"
                        )
                    )
                else:
                    logging.debug(
                        f"Success on attempt {attempt_count} to access {recordingFileFullPath}!"
                    )
                # Process the recording now that the file is available
                newRecording.processRecordingToScopeRuns(
                    leafTransformNode, airwayZoneSegmentationNode, "airwayZone"
                )
                logging.debug(
                    f"Processed new recording to {len(newRecording.listOfScopeRuns)} runs"
                )
                # Update the dropdown list of runs to review
                self.updateRunsToReview()
                # Resave the session file (updated with recording and run data)
                self.currentSession.saveToFile()

    def updateRunsToReview(self):
        """From the current session object, update the dropdown"""
        # Remove all items
        self.runToReviewComboBox.clear()
        scopeRuns = self.currentSession.getListOfScopeRuns()
        scopeRunDict = dict()
        for idx, S in enumerate(scopeRuns):
            key = f"Run{idx}"
            scopeRunDict[key] = S
        self.scopeRunDict = scopeRunDict
        if len(scopeRuns) > 0:
            for runKey in scopeRunDict.keys():
                self.runToReviewComboBox.addItem(runKey)
        else:
            self.runToReviewComboBox.addItem("*no runs recorded this session*")

    def setupScene(self):  # applet specific
        logging.debug("ExampleGuideletGuidelet.setupScene")

        """
        ReferenceToRas transform is used in almost all IGT applications. Reference is the coordinate system
        of a tool fixed to the patient. Tools are tracked relative to Reference, to compensate for patient
        motion. ReferenceToRas makes sure that everything is displayed in an anatomical coordinate system, i.e.
        R, A, and S (Right, Anterior, and Superior) directions in Slicer are correct relative to any
        images or tracked tools displayed.
        ReferenceToRas is needed for initialization, so we need to set it up before calling Guidelet.setupScene().
        """

        try:
            self.referenceToRas = slicer.util.getNode("EmTrackerToHeadSenso")
        except slicer.util.MRMLNodeNotFoundException:
            self.referenceToRas = None
        ## self.referenceToRas = slicer.util.getNode('ReferenceToRas')
        if not self.referenceToRas:
            self.referenceToRas = slicer.vtkMRMLLinearTransformNode()
            self.referenceToRas.SetName("ReferenceToRas")
            m = self.logic.readTransformFromSettings(
                "ReferenceToRas", self.configurationName
            )
            if m is None:
                m = self.logic.createMatrixFromString("1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1")
            self.referenceToRas.SetMatrixTransformToParent(m)
            slicer.mrmlScene.AddNode(self.referenceToRas)

        # Guidelet.setupScene(self) # <--connection with Plus server is made here
        # Guidelet.setupScene just calls AirwayTrackerClass.setupScene, which only sets up the plusRemoteNode
        # (and the reslice driver for the ultrasound version, but we're not using that currently)
        # Moving that code here
        logging.debug(
            "ExampleGuideletGuidelet.setupScene: Getting/Creating PlusRemoteNode and observing"
        )
        self.plusRemoteNode = slicer.mrmlScene.GetFirstNodeByClass(
            "vtkMRMLPlusRemoteNode"
        )
        if self.plusRemoteNode is None:
            self.plusRemoteNode = slicer.vtkMRMLPlusRemoteNode()
            self.plusRemoteNode.SetName("PlusRemoteNode")
            slicer.mrmlScene.AddNode(self.plusRemoteNode)
        self.plusRemoteNode.AddObserver(
            slicer.vtkMRMLPlusRemoteNode.RecordingStartedEvent,
            self.recordingCommandCompleted,
        )
        self.plusRemoteNode.AddObserver(
            slicer.vtkMRMLPlusRemoteNode.RecordingCompletedEvent,
            self.recordingCommandCompleted,
        )
        self.plusRemoteNode.SetAndObserveOpenIGTLinkConnectorNode(self.connectorNode)

        # Not sure why 'EmTrackerToHeadSenso' didn't exist yet, trying processing events here
        slicer.app.processEvents()

        # Which phantom??
        usingJuly9Scan = False
        usingSupineRigid = False
        usingPegNeckHead = False
        usingScannedRigidNeckHead = (
            False  # TODO: make this switchable as a configuration
        )
        using2024PracticeScan = False
        using2024FinalScan = True

        pn = self.parameterNode
        if usingPegNeckHead:
            AIRWAYZONE_SEGMENTATION = PEGNECK_AIRWAYZONE_SEGMENTATION
        elif usingScannedRigidNeckHead:
            AIRWAYZONE_SEGMENTATION = RIGIDNECK_AIRWAYZONE_SEGMENTATION
            # Load matching STL
            outerModelNode = slicer.util.loadModel(RIGIDNECK_STL)
            outerModelNode.GetDisplayNode().SetOpacity(0.1)
        elif usingSupineRigid:
            AIRWAYZONE_SEGMENTATION = SUPINE_AIRWAYZONE_SEGMENTATION
            # Load matching surface
            outerModelNode = slicer.util.loadModel(SUPINE_STL)
            outerModelNode.GetDisplayNode().SetOpacity(0.1)
            # Load matching image
            imageNode = slicer.util.loadVolume(SUPINE_IMAGE)
        elif usingJuly9Scan:
            AIRWAYZONE_SEGMENTATION = JULY9_AIRWAYZONE_SEGMENTATION
            # Load matching surface?
            outerModelNode = slicer.util.loadModel(JULY9_OUTERMODEL_STL)
            # Came from SupineRigid_STL, but manually registered, and then filled
            # in so that the outer layer is an uncomplicated reference with a nose
            # and a closed neck.
            outerModelNode.GetDisplayNode().SetOpacity(0.1)
            # Load matching image
            imageNode = slicer.util.loadVolume(JULY9_IMAGE)
        elif using2024FinalScan:
            imageNode = slicer.util.loadVolume(Final2024_IMAGE)
            AIRWAYZONE_SEGMENTATION = Final2024_AIRWAYZONE_SEGMENTATION
            # Load STL here also (outer model)
            outerModelNode = slicer.util.loadModel(Final2024_OUTERMODEL_STL)
            outerModelNode.GetDisplayNode().SetOpacity(0.1)
            # Load ForViewpoint transform (for bullseye view)
            forViewpointTransform = slicer.util.loadTransform(
                FOR_VIEWPOINT_TRANSFORM_2024_SCOPE1
            )
            self.forViewpointTransform = forViewpointTransform

            # Load sound models (or export from segmentation)
            coughZoneModel = slicer.util.loadModel(COUGH_ZONE_MODEL_STL)
            coughZoneModel.GetDisplayNode().SetColor(COUGH_ZONE_COLOR)
            pn.SetNodeReferenceID("coughZoneModel", coughZoneModel.GetID())
            pn.SetParameter("coughSoundPath", COUGH_SOUND_PATH.as_posix())

            gagZoneModel = slicer.util.loadModel(GAG_ZONE_MODEL_STL)
            gagZoneModel.GetDisplayNode().SetColor(GAG_ZONE_COLOR)
            pn.SetNodeReferenceID("gagZoneModel", gagZoneModel.GetID())
            pn.SetParameter("gagSoundPath", GAG_SOUND_PATH.as_posix())
            septumZoneModel = slicer.util.loadModel(SEPTUM_ZONE_MODEL_STL)
            septumZoneModel.GetDisplayNode().SetColor(SEPTUM_ZONE_COLOR)
            pn.SetNodeReferenceID("septumZoneModel", septumZoneModel.GetID())
            pn.SetParameter("septumSoundPath", SEPTUM_SOUND_PATH.as_posix())

            ouch2ZoneModel = slicer.util.loadModel(OUCH2_ZONE_MODEL_STL)
            ouch2ZoneModel.GetDisplayNode().SetColor(OUCH2_ZONE_COLOR)
            pn.SetNodeReferenceID("ouch2ZoneModel", ouch2ZoneModel.GetID())
            pn.SetParameter("ouch2SoundPath", OUCH2_SOUND_PATH.as_posix())
            #
            # testZoneModel = slicer.util.loadModel(TEST_ZONE_MODEL_STL)
            # pn.SetNodeReferenceID("testZoneModel", testZoneModel.GetID())
            # pn.SetParameter("testZoneSoundPath", TEST_SOUND_PATH.as_posix())
        elif using2024PracticeScan:
            #
            imageNode = slicer.util.loadVolume(AIRWAY_PRACTICE_2024_IMAGE)
            AIRWAYZONE_SEGMENTATION = AIRWAY_PRACTICE_2024_AIRWAYZONE_SEGMENTATION
            # Load STL here also (outer model)

            # Load sound models (or export from segmentation)
            coughZoneModel = slicer.util.loadModel(COUGH_ZONE_MODEL_STL)
            coughZoneModel.GetDisplayNode().SetColor(COUGH_ZONE_COLOR)
            pn.SetNodeReferenceID("coughZoneModel", coughZoneModel.GetID())
            pn.SetParameter("coughSoundPath", COUGH_SOUND_PATH.as_posix())

            gagZoneModel = slicer.util.loadModel(GAG_ZONE_MODEL_STL)
            gagZoneModel.GetDisplayNode().SetColor(GAG_ZONE_COLOR)
            pn.SetNodeReferenceID("gagZoneModel", gagZoneModel.GetID())
            pn.SetParameter("gagSoundPath", GAG_SOUND_PATH.as_posix())

            ouchZoneModel = slicer.util.loadModel(OUCH_ZONE_MODEL_STL)
            ouchZoneModel.GetDisplayNode().SetColor(OUCH_ZONE_COLOR)
            pn.SetNodeReferenceID("septumZoneModel", ouchZoneModel.GetID())
            pn.SetParameter("septumSoundPath", OUCH_SOUND_PATH.as_posix())
            #
            testZoneModel = slicer.util.loadModel(TEST_ZONE_MODEL_STL)
            pn.SetNodeReferenceID("testZoneModel", testZoneModel.GetID())
            pn.SetParameter("testZoneSoundPath", TEST_SOUND_PATH.as_posix())

        # Load airwayZone segmentation
        airwayZoneSegmentationNode = slicer.util.loadSegmentation(
            AIRWAYZONE_SEGMENTATION
        )
        self.parameterNode.SetNodeReferenceID(
            "airwayZoneSegmentationNode", airwayZoneSegmentationNode.GetID()
        )
        self.airwayZoneSegmentationNodeSelector.setCurrentNodeID(
            airwayZoneSegmentationNode.GetID()
        )
        # loading segmentation here also buys some more time for the transforms to get fully loaded into the scene
        self.adjustSegmentationDisplay(airwayZoneSegmentationNode)
        # Center the 3D scene so segmentation is visible
        self.center3Dview()

        slicer.app.processEvents()
        # Hide slice view annotations (patient name, scale, color bar, etc.) as they
        # decrease reslicing performance by 20%-100%
        logging.debug("Hide slice view annotations")
        import DataProbe

        dataProbeUtil = DataProbe.DataProbeLib.DataProbeUtil()
        dataProbeParameterNode = dataProbeUtil.getParameterNode()
        dataProbeParameterNode.SetParameter("showSliceViewAnnotations", "0")

        # Transforms

        logging.debug("Gather transforms")

        # Check if expected transforms are available
        try:
            self.EmTrackerToHeadSensor = slicer.util.getNode("EmTrackerToHeadSenso")
        except slicer.util.MRMLNodeNotFoundException:
            # Conclude we are in testing mode for now
            slicer.util.errorDisplay(
                "Expected transform not found, running it test/debug mode!"
            )
            # Create a dummy tip transform named "Extra"
            self.ExtraTransform = self.createTransformNode(
                translationMm=[0, 0, 7.5], transformName="Extra"
            )
            self.parameterNode.SetNodeReferenceID(
                "sceneLeafTransformNode", self.ExtraTransform.GetID()
            )
            self.leafTransformNodeSelector.setCurrentNodeID(self.ExtraTransform.GetID())
            return  # return early since the rest of the method will fail

        self.EmTrackerToHeadSensor = slicer.util.getNode("EmTrackerToHeadSenso")
        self.StylusSensorToEmTracker = slicer.util.getNode("StylusSensorToEmTrac")
        self.StylusTipToStylusSensor = slicer.util.getNode("StylusTipToStylusSen")
        self.NeedleTipToStylusSensor = slicer.util.getNode("NeedleTipToStylusSen")
        if usingPegNeckHead:
            # self.HeadSensorToHeadSTL = slicer.util.getNode('HeadSensorToPegHeadS')
            self.HeadSensorToHeadSTL = slicer.util.getNode("HeadSensorToNewPegHe")
        elif usingScannedRigidNeckHead:
            self.HeadSensorToHeadSTL = slicer.util.getNode("HeadSensorToRigidHea")
        elif usingSupineRigid:
            self.HeadSensorToHeadSTL = slicer.util.getNode("HeadSensorToScan2STL")
        elif usingJuly9Scan:
            self.HeadSensorToHeadSTL = slicer.util.getNode("HeadSensorToJuly9Sca")
        elif using2024PracticeScan:
            # Reuse because registered 2024 practice to this space
            self.HeadSensorToHeadSTL = slicer.util.getNode("HeadSensorTo2024Prac")
        elif using2024FinalScan:
            self.HeadSensorToHeadSTL = slicer.util.getNode("HeadSensorTo2024Fina")
        else:
            self.HeadSensorToHeadSTL = slicer.util.getNode("HeadSensorToHeadSTL")
        try:
            self.ExtraTransform = slicer.util.getNode("Extra")
        except slicer.util.MRMLNodeNotFoundException:
            # Default to 6mm Z-axis offset (sensor is 6mm from tip of scope)
            self.ExtraTransform = self.createTransformNode(
                translationMm=[0, 0, 6], transformName="Extra"
            )

        # Models
        logging.debug("Create models")

        try:
            self.needleModel = slicer.util.getNode("NeedleModel")
        except slicer.util.MRMLNodeNotFoundException:
            self.needleModel = None
        if not self.needleModel:
            self.needleModel = slicer.modules.createmodels.logic().CreateNeedle(
                80, 1.0, 2.5, 0
            )
            self.needleModel.SetName("NeedleModel")

        # Build transform tree
        logging.debug("Set up transform tree")
        ## In our case, the transform tree is
        ## HeadSensorToHeadSTL > EmTrackerToHeadSenso > StylusSensorToEmTrac > StylusTipToStylusSen
        self.EmTrackerToHeadSensor.SetAndObserveTransformNodeID(
            self.HeadSensorToHeadSTL.GetID()
        )
        self.StylusSensorToEmTracker.SetAndObserveTransformNodeID(
            self.EmTrackerToHeadSensor.GetID()
        )
        # self.StylusTipToStylusSensor.SetAndObserveTransformNodeID(self.StylusSensorToEmTracker.GetID())
        self.NeedleTipToStylusSensor.SetAndObserveTransformNodeID(
            self.StylusSensorToEmTracker.GetID()
        )
        # NOTE Choose one of the following two lines depending on which stylus/sensor type is appropriate
        usingScopeSensor = True  # TODO: don't hard code this
        if usingScopeSensor:
            self.ExtraTransform.SetAndObserveTransformNodeID(
                self.NeedleTipToStylusSensor.GetID()
            )
            self.needleModel.SetAndObserveTransformNodeID(self.ExtraTransform.GetID())
        else:
            # Using stylus sensor (plastic)
            self.needleModel.SetAndObserveTransformNodeID(
                self.StylusTipToStylusSensor.GetID()
            )

        ## self.needleToReference.SetAndObserveTransformNodeID(self.referenceToRas.GetID())
        ## self.needleTipToNeedle.SetAndObserveTransformNodeID(self.needleToReference.GetID())
        ## self.needleModel.SetAndObserveTransformNodeID(self.needleTipToNeedle.GetID())

        # Set "Extra" as default leaf node for processing
        self.parameterNode.SetNodeReferenceID(
            "sceneLeafTransformNode", self.ExtraTransform.GetID()
        )
        self.leafTransformNodeSelector.setCurrentNodeID(self.ExtraTransform.GetID())

        # Add forViewpoint transform to the hierarchy if present
        if hasattr(self, "forViewpointTransform"):
            sceneLeafTransformNode = self.parameterNode.GetNodeReference(
                "sceneLeafTransformNode"
            )
            self.forViewpointTransform.SetAndObserveTransformNodeID(
                sceneLeafTransformNode.GetID()
            )

        return

    def adjustSegmentationDisplay(self, airwayZoneSegmentationNode):
        # Set opacity to transparent
        dn = airwayZoneSegmentationNode.GetDisplayNode()
        dn.SetOpacity(0.2)
        seg = airwayZoneSegmentationNode.GetSegmentation()
        # Set airwayZone as visible but totally transparent
        airwayZoneSegmentID = seg.GetSegmentIdBySegmentName("airwayZone")
        dn.SetSegmentVisibility(airwayZoneSegmentID, True)
        dn.SetSegmentOpacity(airwayZoneSegmentID, 0)
        # Set AirwayLumen as visible and opaque
        airwayLumenSegmentID = seg.GetSegmentIdBySegmentName("AirwayLumen")
        dn.SetSegmentVisibility(airwayLumenSegmentID, True)
        dn.SetSegmentOpacity(airwayLumenSegmentID, 1)
        # Set outer surface as visible but almost totally transparent
        outerSegSegmentID = seg.GetSegmentIdBySegmentName(
            "Rigid Sinus Model_FullyAssembled"
        )  # name for pegneck segmentation, no corresponding segment for RigidNeck
        if not outerSegSegmentID == "":
            dn.SetSegmentVisibility(outerSegSegmentID, True)
            dn.SetSegmentOpacity(
                outerSegSegmentID, 0.25
            )  # multiplied by the overall opacity
        # Set all other segments as not visible
        for idx in range(seg.GetNumberOfSegments()):
            segID = seg.GetNthSegmentID(idx)
            if segID not in [
                airwayZoneSegmentID,
                airwayLumenSegmentID,
                outerSegSegmentID,
            ]:
                dn.SetSegmentVisibility(segID, False)

    def center3Dview(self):
        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0)
        threeDView = threeDWidget.threeDView()
        threeDView.resetFocalPoint()
        # threeDView.rotateToViewAxis(4) # also rotate so looking at face for RigidNeck model

    def recordingCommandCompleted(self, command, q):
        """lifted from AirwayTrackerClass.py"""
        statusText = "Recording "
        statusText = (
            statusText
            + self.plusRemoteNode.GetRecordingStatusAsString(
                self.plusRemoteNode.GetRecordingStatus()
            )
            + " "
        )
        statusText = statusText + self.plusRemoteNode.GetRecordingMessage() + " "
        logging.info(statusText)
        self.startStopRecordingButton.setToolTip(statusText)

    def createTransformNode(
        self, translationMm=(0, 0, 0), transformName="CreatedTransform"
    ):
        """Create a simple translation-only linear transform node from scratch"""
        import numpy as np

        transformNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLinearTransformNode", transformName
        )
        transformMatrix = np.eye(4, dtype=float)
        transformMatrix[0:3, 3] = translationMm
        transformNode.SetAndObserveMatrixTransformToParent(
            slicer.util.vtkMatrixFromArray(transformMatrix)
        )
        return transformNode

    def disconnect(self):  # TODO see connect
        logging.debug("ExampleGuideletGuidelet.disconnect()")
        Guidelet.disconnect(self)
        # Disconnect buttons (moved from AirwayTrackerClass.preCleanup -> disconnect)
        self.startStopRecordingButton.disconnect(
            "clicked(bool)", self.onStartStopRecordingClicked
        )
        self.userNameLineEdit.disconnect(
            "editingFinished()", self.updateParameterNodeFromGuideletGUI
        )
        self.roleComboBox.disconnect(
            "currentIndexChanged(int)", self.updateParameterNodeFromGuideletGUI
        )
        self.experienceLevelComboBox.disconnect(
            "currentIndexChanged(int)", self.updateParameterNodeFromGuideletGUI
        )
        self.saveUserInfoButton.disconnect(
            "clicked(bool)", self.saveUserInfoButtonClicked
        )
        self.displaySelectedRunButton.disconnect(
            "clicked(bool)", self.onDisplaySelectedRunClicked
        )
        self.liveUpdateCheckBox.disconnect(
            "toggled(bool)", self.onLiveUpdateCheckBoxToggled
        )

    def patientSetupPanel(self):
        logging.debug("patientSetupPanel")

        # Load from UI file
        moduleDir = os.path.dirname(__file__)
        uiFilePath = os.path.join(moduleDir, "Resources", "UI", "TrackerUIMike.ui")
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
        self.expertRunToCompareLabel = loadedUI.ExpertRunToCompareLabel
        self.runToReviewLabel = loadedUI.RunToReviewLabel
        self.liveUpdateCheckBox = loadedUI.LiveUpdateCheckBox

        #### TEMPORARY CHANGES ####
        self.displaySelectedRunButton.setText("Display Selected Run")  # instead of runs
        self.expertRunToCompareComboBox.hide()
        self.expertRunToCompareLabel.hide()
        self.experienceLevelComboBox.hide()
        loadedUI.label_2.hide()  # experience label
        self.roleComboBox.hide()
        loadedUI.label_3.hide()  # role label
        self.currentExperienceLevelLabel.hide()
        self.currentRoleLabel.hide()
        loadedUI.label_5.hide()  # current exp label
        loadedUI.label_6.hide()  # current role label

    def onExampleButtonClicked(self, toggled):
        logging.debug("onExampleButtonClicked")

    def onPatientSetupPanelToggled(self, toggled):
        if toggled == False:
            return
        logging.debug("onPatientSetupPanelToggled: {0}".format(toggled))
        # self.selectView(self.VIEW_ULTRASOUND_3D)

    def onUltrasoundPanelToggled(self, toggled):
        if not toggled:
            # deactivate placement mode
            interactionNode = slicer.app.applicationLogic().GetInteractionNode()
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)
            return

        logging.debug("onUltrasoundPanelToggled: {0}".format(toggled))

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
        if hasattr(
            slicer.vtkMRMLViewNode(), "SetOrientationMarkerType"
        ):  # orientation marker is not available in older Slicer versions
            v1 = slicer.util.getNode("View1")
            v1.SetOrientationMarkerType(v1.OrientationMarkerTypeNone)
