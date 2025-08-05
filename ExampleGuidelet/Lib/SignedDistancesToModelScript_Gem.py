import slicer
import vtk
import numpy as np
import math

# --- Configuration ---
# Replace 'MyModelNode' with the actual name of your model node in Slicer
# You can find the name in the Data module or by clicking on the model in the scene.
MODEL_NODE_NAME = "Side_SeptumZone"  # e.g., 'Sphere', 'Skull', 'Femur', etc.
model_node = getNode(MODEL_NODE_NAME)

# Define your array of 3D positions (x, y, z)
# Example: A few points around the origin
input_positions = np.array(
    [[0.0, 0.0, 0.0], [10.0, 5.0, -2.0], [-7.0, 1.0, 8.0], [0.1, 0.2, 0.3]]
)


# --- Function to calculate signed distances ---
def get_signed_distances_to_model(model_node, positions_array):
    """
    Calculates the signed distance from an array of 3D positions to the closest point
    on a specified vtkMRMLModelNode in 3D Slicer.
    Positive distance means outside, negative means inside, zero means on the surface.

    Args:
        model_node (vtkMRMLModelNode): The vtkMRMLModelNode in the Slicer scene.
        positions_array (np.ndarray): A NumPy array of shape (N, 3) where N is
                                      the number of points, and each row is a
                                      [x, y, z] coordinate.

    Returns:
        list: A list of tuples (signed_distance, closest_point_coords), one for each input position.
              Returns an empty list if the model node is not found or has no polydata.
    """
    # 1. Get the vtkMRMLModelNode
    # model_node = slicer.util.getNode(model_node_name)
    # if not model_node:
    #    print(f"Error: Model node '{model_node_name}' not found in the scene.")
    #    return []
    model_node_name = model_node.GetName()

    # 2. Get the vtkPolyData from the model node
    poly_data = model_node.GetPolyData()
    if not poly_data or poly_data.GetNumberOfPoints() == 0:
        print(f"Error: Model node '{model_node_name}' has no valid polydata.")
        return []

    # 3. Create and set up vtkImplicitPolyDataDistance
    print(f"Setting up vtkImplicitPolyDataDistance for model '{model_node_name}'...")
    implicit_distance = vtk.vtkImplicitPolyDataDistance()
    implicit_distance.SetInput(poly_data)
    # Note: vtkImplicitPolyDataDistance internally builds a locator for efficiency.
    print("vtkImplicitPolyDataDistance initialized.")

    results = []  # Store (signed_distance, closest_point) tuples

    # Prepare mutable variables for vtkImplicitPolyDataDistance output
    closestPoint = [
        0.0,
        0.0,
        0.0,
    ]  # Array to store the closest point found on the surface

    # 4. Iterate through each input position and find the signed distance
    print(f"Calculating signed distances for {len(positions_array)} points...")
    for i, point in enumerate(positions_array):
        # EvaluateFunctionAndGetClosestPoint returns the signed distance
        # and populates the closestPoint array.
        signed_distance = implicit_distance.EvaluateFunctionAndGetClosestPoint(
            point.tolist(), closestPoint
        )

        results.append((signed_distance, tuple(closestPoint)))
        print(
            f"Point {i}: {point} -> Signed distance: {signed_distance:.3f}, Closest Point: {closestPoint}"
        )

    print("Signed distance calculation complete.")
    return results


# --- Main execution ---
if __name__ == "__main__":
    # Ensure a model exists for testing. If you don't have one, you can
    # uncomment the following lines to create a simple sphere model.
    # This is for demonstration purposes.
    #
    # if not slicer.util.getNode(MODEL_NODE_NAME):
    #     print(f"Creating a dummy sphere model named '{MODEL_NODE_NAME}' for demonstration...")
    #     sphere = vtk.vtkSphereSource()
    #     sphere.SetRadius(5.0) # Adjust radius as needed
    #     sphere.Update()
    #     modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", MODEL_NODE_NAME)
    #     modelNode.SetAndObservePolyData(sphere.GetOutput())
    #     slicer.app.processEvents() # Ensure Slicer updates its scene

    calculated_results = get_signed_distances_to_model(model_node, input_positions)

    if calculated_results:
        print("\n--- Summary of Calculated Signed Distances ---")
        for i, (dist, closest_pt) in enumerate(calculated_results):
            print(
                f"Position {input_positions[i]} -> Signed Distance: {dist:.3f}, Closest Point: {closest_pt}"
            )
    else:
        print(
            "\nNo distances calculated. Please ensure the model node exists and has valid polydata."
        )


import vtk
import slicer
from slicer.util import getNode


def visualizeCellNormals(modelNode):
    """
    Computes and visualizes cell normals for a specified 3D Slicer Model node.

    Args:
        modelNodeID (str): The node ID of the Slicer Model (e.g., 'vtkMRMLModelNode1').
                           You can find this in the Data module or by selecting the model
                           and looking at its properties.
    """

    # modelNode = getNode(modelNode)
    if not modelNode:
        slicer.util.errorDisplay(f"Model node not found.")
        return

    polydata = modelNode.GetPolyData()
    if not polydata:
        slicer.util.errorDisplay(
            f"Model node '{modelNode.GetName()}' does not contain polydata."
        )
        return

    # 1. Compute Cell Normals
    normalsFilter = vtk.vtkPolyDataNormals()
    normalsFilter.SetInputData(polydata)
    normalsFilter.SetComputeCellNormals(True)  # Crucial: compute cell normals
    normalsFilter.SetComputePointNormals(False)  # Not needed for this visualization
    # normalsFilter.SetConsistency(True)  # Try to make normals consistent
    # normalsFilter.SetAutoOrientNormals(True)  # Attempt to orient them outwards
    normalsFilter.Update()

    # Get the polydata with computed normals
    polydataWithNormals = normalsFilter.GetOutput()

    # Ensure cell normals exist after processing
    cellNormals = polydataWithNormals.GetCellData().GetNormals()
    if not cellNormals:
        slicer.util.warning(
            "Could not compute cell normals. Check your mesh for issues."
        )
        return
    # The default name for normals generated by vtkPolyDataNormals is "Normals"
    normalArrayName = cellNormals.GetName()
    if not normalArrayName:  # If for some reason it's not named, it's usually "Normals"
        normalArrayName = "Normals"

    # 2. Create Glyphs for Visualization
    glyphSource = vtk.vtkLineSource()  # Use a line to represent the normal
    glyphSource.SetPoint1(0, 0, 0)
    glyphSource.SetPoint2(0, 0.2, 0)  # Adjust length as needed, relative to model size

    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(glyphSource.GetOutputPort())
    glyph.SetInputData(polydataWithNormals)
    glyph.OrientOn()
    # glyph.SetInputArrayToProcess(
    #    0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, normalArrayName
    # )
    glyph.SetVectorModeToUseVector()  # Use cell normals for orientation
    glyph.SetScaleModeToDataScalingOff()  # Don't scale glyphs by data values
    glyph.SetScaleFactor(1.0)  # Adjust this to control the visual length of the normals
    # Relative to the glyphSource.SetPoint2 length
    glyph.Update()

    # 3. Add Glyphs as a New Model Node in Slicer
    outputModelNode = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLModelNode", modelNode.GetName() + "_CellNormals"
    )
    outputModelNode.SetAndObservePolyData(glyph.GetOutput())

    # Optional: Set display properties for the normal glyphs
    outputModelNode.CreateDefaultDisplayNodes()
    displayNode = outputModelNode.GetDisplayNode()
    if displayNode:
        displayNode.SetColor(1, 0, 0)  # Red color for normals
        displayNode.SetOpacity(1.0)
        displayNode.SetVisibility(True)
        # For lines, you might want to adjust line thickness (though not directly on displayNode for models)
        # You'll see individual lines for each normal.

    slicer.util.infoDisplay(
        f"Cell normals visualized for '{modelNode.GetName()}' as '{outputModelNode.GetName()}'"
    )
    # slicer.app.layoutManager().fitViewInAllLayouts()
    return glyph


# --- How to use this script in 3D Slicer ---

# 1. Load your 3D model into Slicer (e.g., File -> Add Data, or Drag and Drop)
# 2. Open the Python Interactor (View -> Python Interactor)
# 3. Copy and paste the 'visualizeCellNormals' function definition into the Interactor and press Enter.
# 4. Find the Node ID of your loaded model. You can:
#    a. Go to the "Data" module.
#    b. Expand the "Models" section.
#    c. Find your model (e.g., "MyModel"). The Node ID is usually like "vtkMRMLModelNode1", "vtkMRMLModelNode2", etc.
#       You can also right-click on the model in the Data module and select "Copy Node ID".
# 5. Call the function with your model's Node ID:
#    For example, if your model's Node ID is 'vtkMRMLModelNode1':
#    visualizeCellNormals('vtkMRMLModelNode1')
#
#    If you want to quickly get the current active model:
#    activeModelNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLModelNode") # Gets the first model, might not be what you want
#    if activeModelNode:
#        print(f"Active model node ID: {activeModelNode.GetID()}")
#        visualizeCellNormals(activeModelNode.GetID())
#    else:
#        print("No model nodes found in the scene.")

# --- Example Usage (Uncomment one of these lines after loading your model) ---
# visualizeCellNormals('vtkMRMLModelNode1') # Replace with your actual model Node ID
# visualizeCellNormals('MyModel_1') # If you renamed your model, its ID might be different
