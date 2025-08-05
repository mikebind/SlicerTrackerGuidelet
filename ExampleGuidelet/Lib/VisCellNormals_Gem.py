import vtk
import slicer


def visualizeCellNormalsRobust(modelNode):
    """
    Computes and robustly visualizes cell normals for a specified 3D Slicer Model node,
    ensuring glyphs are oriented correctly with cell normals.

    Args:
        modelNodeID (str): The node ID of the Slicer Model (e.g., 'vtkMRMLModelNode1').
    """

    modelNodeID = modelNode.GetID()
    if not modelNode:
        slicer.util.error(f"Model node '{modelNodeID}' not found.")
        return

    polydata = modelNode.GetPolyData()
    if not polydata:
        slicer.util.error(f"Model node '{modelNodeID}' does not contain polydata.")
        return

    # --- 1. Compute Cell Normals ---
    normalsFilter = vtk.vtkPolyDataNormals()
    normalsFilter.SetInputData(polydata)
    normalsFilter.SetComputeCellNormals(True)
    normalsFilter.SetComputePointNormals(False)  # Crucial: only compute cell normals
    normalsFilter.SetConsistency(True)
    normalsFilter.SetAutoOrientNormals(True)
    normalsFilter.Update()

    polydataWithNormals = normalsFilter.GetOutput()

    cellNormalsArray = polydataWithNormals.GetCellData().GetNormals()
    if not cellNormalsArray:
        slicer.util.warning(
            "Failed to compute cell normals. No 'Normals' array found in CellData."
        )
        return

    normalArrayName = cellNormalsArray.GetName()
    if not normalArrayName:
        # vtkPolyDataNormals typically names it "Normals"
        normalArrayName = "Normals"
        cellNormalsArray.SetName(normalArrayName)  # Ensure it has a name if not set

    # --- 2. Prepare Data for Glyphs (Cell Centers and Normals in PointData) ---
    # Glyphs usually operate on PointData. To display cell normals at cell centers,
    # we need to create points at cell centers and associate the normals with these new points.

    # Create points at cell centroids
    cellCenters = vtk.vtkCellCenters()
    cellCenters.SetInputData(polydataWithNormals)
    cellCenters.VertexCellsOn()  # Generate a vertex for each cell center
    cellCenters.Update()

    # Get the output polydata from cellCenters (which now has points at centroids)
    polydataForGlyphs = cellCenters.GetOutput()

    # Now, copy the cell normals from polydataWithNormals to the PointData of polydataForGlyphs
    # This is critical because glyphs expect normals in PointData if they are to be placed at points.
    polydataForGlyphs.GetPointData().SetNormals(cellNormalsArray)

    # --- 3. Create Glyphs for Visualization ---
    glyphSource = vtk.vtkLineSource()
    glyphSource.SetPoint1(0, 0, 0)
    glyphLength = 0.2  # Adjust this visually as needed
    glyphSource.SetPoint2(glyphLength, 0, 0)

    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(glyphSource.GetOutputPort())
    glyph.SetInputData(
        polydataForGlyphs
    )  # Input is now the polydata with centroids and normals in PointData

    # Set the array to use. Now we're looking for NORMALS in POINT_DATA.
    # glyph.SetInputArrayToProcess(
    #    0,  # internal vtkAlgorithm index
    #    0,  # Port index (usually 0)
    #    vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,  # IMPORTANT: Now we associate with points
    #    vtk.vtkDataSetAttributes.NORMALS,  # Specify that it's a normal attribute
    #    normalArrayName,  # The string name of the array (now copied to PointData)
    # )

    # Set the vector mode to use the vector from the specified array
    # glyph.SetVectorModeToUseVector()
    glyph.SetVectorModeToUseNormal()

    # Scale glyphs by the magnitude of the vector (normals are unit length, so this just applies SetScaleFactor)
    glyph.SetScaleModeToScaleByVector()
    glyph.SetScaleFactor(1.0)  # Apply this factor to the glyphSource.SetPoint2 length

    glyph.Update()

    # --- 4. Add Glyphs as a New Model Node in Slicer ---
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
        # displayNode.SetPointSize(2)  # Sometimes helps if glyphs are small dots

    slicer.util.infoDisplay(
        f"Cell normals visualized for '{modelNodeID}' as '{outputModelNode.GetName()}'"
    )
    # slicer.app.layoutManager().fitViewInAllLayouts()
    return glyph


import vtk
import slicer


### DOESN"T WORK:
def identifyCellsNearPoint(modelNode, queryPoint, searchRadius=0.1):
    """
    Identifies and prints information about cells whose centroids are near a query point.

    Args:
        modelNodeID (str): The node ID of the Slicer Model.
        queryPoint (list or tuple): [x, y, z] coordinates of the point of interest.
        searchRadius (float): Max distance from centroid to queryPoint to be considered "near".
    """
    modelNodeID = modelNode.GetID()
    if not modelNode:
        slicer.util.error(f"Model node '{modelNodeID}' not found.")
        return

    polydata = modelNode.GetPolyData()
    if not polydata:
        slicer.util.error(f"Model node '{modelNodeID}' does not contain polydata.")
        return

    print(f"\n--- Investigating cells near {queryPoint} with radius {searchRadius} ---")

    # Get the cell locator (from the implicit distance function or create a new one)
    # It's good practice to build it once.
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(polydata)
    cellLocator.BuildLocator()

    # Get a list of all cell IDs within the search radius
    cellIds = vtk.vtkIdList()
    cellLocator.FindCellsWithinRadius(queryPoint, searchRadius, cellIds)

    if cellIds.GetNumberOfIds() == 0:
        print(f"No cells found within {searchRadius} of {queryPoint}.")
        return

    print(f"Found {cellIds.GetNumberOfIds()} cells near {queryPoint}:")

    cellPoints = vtk.vtkIdList()
    cellNormal = [0.0, 0.0, 0.0]
    p1 = [0.0, 0.0, 0.0]
    p2 = [0.0, 0.0, 0.0]
    p3 = [0.0, 0.0, 0.0]

    for i in range(cellIds.GetNumberOfIds()):
        cellId = cellIds.GetId(i)
        cell = polydata.GetCell(cellId)

        if cell.GetCellType() != vtk.VTK_TRIANGLE:
            print(
                f"  Cell ID {cellId}: Not a triangle (Type: {cell.GetCellType()}). Skipping normal check."
            )
            continue

        # Get the triangle vertices
        polydata.GetPoint(cell.GetPointId(0), p1)
        polydata.GetPoint(cell.GetPointId(1), p2)
        polydata.GetPoint(cell.GetPointId(2), p3)

        # Calculate the normal for this specific triangle
        vtk.vtkTriangle.ComputeNormal(p1, p2, p3, cellNormal)
        vtk.vtkMath.Normalize(cellNormal)

        # Get cell centroid (for comparison, though not strictly needed here)
        center = [0.0, 0.0, 0.0]
        numPts = cell.GetNumberOfPoints()
        for k in range(numPts):
            pt = polydata.GetPoint(cell.GetPointId(k))
            center[0] += pt[0]
            center[1] += pt[1]
            center[2] += pt[2]
        center[0] /= numPts
        center[1] /= numPts
        center[2] /= numPts

        print(f"  Cell ID {cellId}:")
        print(
            f"    Vertices: {polydata.GetPoint(cell.GetPointId(0))}, {polydata.GetPoint(cell.GetPointId(1))}, {polydata.GetPoint(cell.GetPointId(2))}"
        )
        print(
            f"    Normal: [{cellNormal[0]:.4f}, {cellNormal[1]:.4f}, {cellNormal[2]:.4f}]"
        )
        print(f"    Centroid: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
        print("-" * 30)

    # Optional: Create a new model containing only these problematic cells for isolated viewing
    problem_cells_polydata = vtk.vtkPolyData()
    problem_cells_polydata.SetPoints(polydata.GetPoints())

    problem_polys = vtk.vtkCellArray()
    for i in range(cellIds.GetNumberOfIds()):
        cellId = cellIds.GetId(i)
        cell = polydata.GetCell(cellId)
        if cell.GetCellType() == vtk.VTK_TRIANGLE:  # Only copy triangles
            # Create a new triangle cell and add its points
            tri = vtk.vtkTriangle()
            for j in range(3):
                tri.GetPointIds().SetId(j, cell.GetPointId(j))
            problem_polys.InsertNextCell(tri)

    problem_cells_polydata.SetPolys(problem_polys)

    if problem_cells_polydata.GetNumberOfCells() > 0:
        outputModelNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", modelNode.GetName() + "_ProblemCells"
        )
        outputModelNode.SetAndObservePolyData(problem_cells_polydata)
        displayNode = outputModelNode.GetDisplayNode()
        if displayNode:
            displayNode.SetColor(0, 1, 1)  # Cyan color
            displayNode.SetOpacity(0.8)
            displayNode.SetRepresentationToSurface()
            displayNode.SetEdgeVisibility(1)  # Show edges to see individual triangles
        slicer.util.info(
            f"Problematic cells extracted to '{outputModelNode.GetName()}'"
        )


# --- Usage Example ---
# 1. Load your model in Slicer.
# 2. Find a problematic area visually (where normals overlap).
# 3. Use the "Markups" module to place a fiducial (Ctrl+Click in 3D view) at the center of the problematic region.
# 4. Read the coordinates of that fiducial from the Markups table.
# 5. Replace 'YourModelNodeID' and the coordinates below:

# model_id = 'vtkMRMLModelNode1' # Replace with your model's actual ID
# problematic_center = [10.0, 5.0, 2.0] # Replace with coordinates from your fiducial
# search_radius = 0.5 # Adjust based on how tightly packed the problematic cells are

# identifyCellsNearPoint(model_id, problematic_center, search_radius)

import vtk
import slicer


def visualizeNonManifoldEdges(modelNode):
    """
    Extracts and visualizes non-manifold edges of a Slicer Model node.
    """
    modelNodeID = modelNode.GetID()
    if not modelNode:
        slicer.util.error(f"Model node '{modelNodeID}' not found.")
        return

    polydata = modelNode.GetPolyData()
    if not polydata:
        slicer.util.error(f"Model node '{modelNodeID}' does not contain polydata.")
        return

    featureEdges = vtk.vtkFeatureEdges()
    featureEdges.SetInputData(polydata)
    featureEdges.SetBoundaryEdges(False)  # Don't show open boundaries
    featureEdges.SetFeatureEdges(False)  # Don't show sharp creases
    featureEdges.SetManifoldEdges(False)  # Don't show regular edges (shared by 2 cells)
    featureEdges.SetNonManifoldEdges(
        True
    )  # *** Show edges shared by >2 cells (or <2, but vtk will filter those differently)
    featureEdges.Update()

    nonManifoldPolyData = featureEdges.GetOutput()

    if nonManifoldPolyData.GetNumberOfPoints() == 0:
        slicer.util.infoDisplay(f"No non-manifold edges found for '{modelNodeID}'.")
        return

    outputModelNode = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLModelNode", modelNode.GetName() + "_NonManifoldEdges"
    )
    outputModelNode.SetAndObservePolyData(nonManifoldPolyData)

    displayNode = outputModelNode.GetDisplayNode()
    if displayNode:
        displayNode.SetColor(1, 0, 1)  # Magenta color for non-manifold edges
        displayNode.SetOpacity(1.0)
        displayNode.SetRepresentationToWireframe()  # Show as lines
        displayNode.SetPointSize(5)  # Make points visible if they are isolated
        displayNode.SetLineWidth(3)  # Make lines thicker

    slicer.util.infoDisplay(
        f"Non-manifold edges visualized for '{modelNodeID}' as '{outputModelNode.GetName()}' (magenta)."
    )
    # slicer.app.layoutManager().fitViewInAllLayouts()


# --- Usage Example ---
# model_id = 'vtkMRMLModelNode1' # Replace with your model's actual ID
# visualizeNonManifoldEdges(model_id)


def identifyCellsNearPoint(modelNode, queryPoint, searchRadius=0.5):
    """
    Identifies and prints information about cells whose centroids are near a query point.
    It also creates a new model containing only these cells for isolated viewing.

    Args:
        modelNode (vtkMRMLModelNode): The Slicer Model node itself.
        queryPoint (list or tuple): [x, y, z] coordinates of the point of interest.
        searchRadius (float): Max distance from cell centroid to queryPoint to be considered "near".
                              Adjust this based on your model's scale.
    """
    if not modelNode:
        slicer.util.errorDisplay("Input model node is None.")
        return

    polydata = modelNode.GetPolyData()
    if not polydata:
        slicer.util.errorDisplay(
            f"Model node '{modelNode.GetName()}' does not contain polydata."
        )
        return

    modelNodeID = (
        modelNode.GetID()
    )  # Get ID if needed for messages, but not used as primary input

    print(f"\n--- Investigating cells near {queryPoint} with radius {searchRadius} ---")

    found_cell_ids = vtk.vtkIdList()

    p1 = [0.0, 0.0, 0.0]
    p2 = [0.0, 0.0, 0.0]
    p3 = [0.0, 0.0, 0.0]

    num_cells = polydata.GetNumberOfCells()
    slicer.util.infoDisplay(
        f"Checking {num_cells} cells in model '{modelNode.GetName()}'..."
    )

    for cellId in range(num_cells):
        cell = polydata.GetCell(cellId)

        # Only process triangles for normal calculation relevance
        if cell.GetCellType() != vtk.VTK_TRIANGLE:
            # You might want to remove this print for performance on large models
            # print(f"  Cell ID {cellId}: Not a triangle (Type: {cell.GetCellType()}). Skipping.")
            continue

        # Calculate cell centroid
        center = [0.0, 0.0, 0.0]
        num_points_in_cell = cell.GetNumberOfPoints()

        # Handle degenerate cells that might have no points or less than 3 for a triangle
        if num_points_in_cell < 3:  # A triangle must have 3 points
            continue

        for k in range(num_points_in_cell):
            pt_id = cell.GetPointId(k)
            pt_coords = polydata.GetPoint(pt_id)
            center[0] += pt_coords[0]
            center[1] += pt_coords[1]
            center[2] += pt_coords[2]

        center[0] /= num_points_in_cell
        center[1] /= num_points_in_cell
        center[2] /= num_points_in_cell

        # Calculate distance from cell centroid to query point
        distance_sq = vtk.vtkMath.Distance2BetweenPoints(queryPoint, center)

        if distance_sq <= searchRadius**2:
            found_cell_ids.InsertNextId(cellId)

            # Print cell details
            cellNormal = [0.0, 0.0, 0.0]

            # Ensure we have 3 points for normal calculation
            if num_points_in_cell >= 3:
                polydata.GetPoint(cell.GetPointId(0), p1)
                polydata.GetPoint(cell.GetPointId(1), p2)
                polydata.GetPoint(cell.GetPointId(2), p3)
                vtk.vtkTriangle.ComputeNormal(p1, p2, p3, cellNormal)
                vtk.vtkMath.Normalize(cellNormal)
            else:
                cellNormal = [
                    float("nan"),
                    float("nan"),
                    float("nan"),
                ]  # Indicate invalid normal for degenerate cell

            print(f"  Cell ID {cellId}:")
            print(
                f"    Normal: [{cellNormal[0]:.4f}, {cellNormal[1]:.4f}, {cellNormal[2]:.4f}]"
            )
            print(f"    Centroid: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
            print("-" * 30)

    if found_cell_ids.GetNumberOfIds() == 0:
        slicer.util.infoDisplay(
            f"No cells found within {searchRadius} of {queryPoint} based on centroid distance."
        )
        return

    slicer.util.infoDisplay(
        f"Total problematic cells identified: {found_cell_ids.GetNumberOfIds()}"
    )

    # Create a new model containing only these problematic cells for isolated viewing
    problem_cells_polydata = vtk.vtkPolyData()
    # Deep copy the points if you want them isolated from the original,
    # otherwise, referencing the original points is fine for display
    problem_cells_polydata.SetPoints(polydata.GetPoints())

    problem_polys = vtk.vtkCellArray()
    for i in range(found_cell_ids.GetNumberOfIds()):
        cellId = found_cell_ids.GetId(i)
        cell = polydata.GetCell(cellId)

        # Only add triangles to the new polydata
        if cell.GetCellType() == vtk.VTK_TRIANGLE and cell.GetNumberOfPoints() == 3:
            tri = vtk.vtkTriangle()
            for j in range(3):
                tri.GetPointIds().SetId(j, cell.GetPointId(j))
            problem_polys.InsertNextCell(tri)

    problem_cells_polydata.SetPolys(problem_polys)

    if problem_cells_polydata.GetNumberOfCells() > 0:
        outputModelNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", modelNode.GetName() + "_ProblemCells"
        )
        outputModelNode.SetAndObservePolyData(problem_cells_polydata)
        displayNode = outputModelNode.GetDisplayNode()
        if displayNode:
            displayNode.SetColor(0, 1, 1)  # Cyan color
            displayNode.SetOpacity(0.8)
            displayNode.SetRepresentationToSurface()
            displayNode.SetEdgeVisibility(1)  # Show edges to see individual triangles
            displayNode.SetPointSize(2)  # Show vertex points
            displayNode.SetPointVisibility(1)
        slicer.util.infoDisplay(
            f"Problematic cells extracted to '{outputModelNode.GetName()}'"
        )
    else:
        slicer.util.infoDisplay(
            "No valid triangle cells found to extract for the problem area."
        )


# --- How to use this script in 3D Slicer ---

# 1. Load your 3D model into Slicer.
# 2. Open the Python Interactor (View -> Python Interactor).
# 3. Copy and paste the 'identifyCellsNearPoint' function definition.
# 4. Get a reference to your loaded model node.
#    You can select it in the Data module and use `slicer.util.getNode('YourModelNodeNameOrID')`.
#    Or use `slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLModelNode")` if only one model.
# 5. Place a Markup Fiducial (Ctrl+Click in 3D view) at the center of the problematic region.
# 6. Read the coordinates of that fiducial from the Markups table.
# 7. Call the function:

# Example:
# model_node = slicer.util.getNode('vtkMRMLModelNode1') # Replace with your actual model node or its ID
# problematic_center = [10.0, 5.0, 2.0] # Replace with coordinates from your fiducial
# search_radius = 0.5 # Adjust as needed

# if model_node:
#     identifyCellsNearPoint(model_node, problematic_center, search_radius)
# else:
#     slicer.util.errorDisplay("Could not get a reference to the model node.")


import vtk
import slicer
from slicer.util import getNode, infoDisplay, errorDisplay


def analyzeCoincidentPoints(
    modelNode, tolerance=1e-6
):  # Default tolerance for floating point comparisons
    """
    Analyzes a model for coincident points within a specified tolerance
    and reports groups of such points.

    Args:
        modelNode (vtkMRMLModelNode): The Slicer Model node.
        tolerance (float): The maximum distance between points to be considered coincident.
                           Adjust this based on your model's scale. For millimeter scale,
                           1e-6 (1 micron) is a good starting point.
    """
    if not modelNode:
        errorDisplay("Input model node is None.")
        return

    polydata = modelNode.GetPolyData()
    if not polydata:
        errorDisplay(f"Model node '{modelNode.GetName()}' does not contain polydata.")
        return

    infoDisplay(
        f"Analyzing '{modelNode.GetName()}' for coincident points with tolerance {tolerance}..."
    )

    coincidentPoints = vtk.vtkCoincidentPoints()
    coincidentPoints.SetInputPoints(polydata.GetPoints())
    coincidentPoints.SetTolerance(tolerance)
    coincidentPoints.BuildLocator()  # This builds the internal data structure

    # Iterate through groups of coincident points
    coincidentPoints.InitTraversal()

    group_count = 0
    total_coincident_points = 0

    print(
        f"\n--- Coincident Point Analysis for '{modelNode.GetName()}' (Tolerance: {tolerance}) ---"
    )

    while True:
        coincident_ids = coincidentPoints.GetNextCoincidentPointIds()
        if coincident_ids is None:
            break  # No more groups

        num_coincident = coincident_ids.GetNumberOfIds()
        if num_coincident > 1:
            group_count += 1
            total_coincident_points += num_coincident

            # Get the coordinates of the first point in the group (they are all coincident)
            first_point_id = coincident_ids.GetId(0)
            coords = polydata.GetPoint(first_point_id)

            print(
                f"Group {group_count}: {num_coincident} coincident points at approx. [{coords[0]:.6f}, {coords[1]:.6f}, {coords[2]:.6f}]"
            )
            print(
                f"  Point IDs: {[coincident_ids.GetId(i) for i in range(num_coincident)]}"
            )

    if group_count == 0:
        infoDisplay(
            f"No groups of coincident points found within tolerance {tolerance}."
        )
    else:
        infoDisplay(
            f"Found {group_count} groups with a total of {total_coincident_points} coincident points."
        )
        print(f"Total points in model: {polydata.GetNumberOfPoints()}")
        print(f"Total groups of coincident points: {group_count}")
        print(f"Total points involved in coincidence: {total_coincident_points}")

    print("-" * 60)


# --- Usage Example ---
# 1. Load your model.
# 2. Get a reference to your model node (e.g., from `slicer.util.getNode('vtkMRMLModelNode1')`).
# 3. Call the function:

# model_node = slicer.util.getNode('vtkMRMLModelNode1') # !!! Replace with your model node !!!
# if model_node:
#     # Start with a very small tolerance. If nothing found, gradually increase it.
#     analyzeCoincidentPoints(model_node, tolerance=1e-7) # e.g., 0.1 microns
#     # If still nothing, try 1e-6, 1e-5, etc., depending on your model's scale
# else:
#     errorDisplay("Model node not found.")

import vtk
import slicer
from slicer.util import getNode, infoDisplay, errorDisplay


def analyzeDegenerateTriangles(
    modelNode, min_area_threshold=1e-9, max_edge_ratio=100.0
):
    """
    Analyzes a model for degenerate or very small triangles.

    Args:
        modelNode (vtkMRMLModelNode): The Slicer Model node.
        min_area_threshold (float): Triangles with area below this will be flagged as too small.
                                    Adjust based on model scale (e.g., 1e-9 mm^2 for sub-micron area).
        max_edge_ratio (float): A measure of triangle "sliveriness". If the ratio of the longest
                                edge to the shortest edge is above this, it's considered degenerate.
    """
    if not modelNode:
        errorDisplay("Input model node is None.")
        return

    polydata = modelNode.GetPolyData()
    if not polydata:
        errorDisplay(f"Model node '{modelNode.GetName()}' does not contain polydata.")
        return

    infoDisplay(f"Analyzing '{modelNode.GetName()}' for degenerate/small triangles...")

    degenerate_cell_ids = vtk.vtkIdList()

    p1 = [0.0, 0.0, 0.0]
    p2 = [0.0, 0.0, 0.0]
    p3 = [0.0, 0.0, 0.0]

    print(f"\n--- Degenerate Triangle Analysis for '{modelNode.GetName()}' ---")
    print(f"  Min Area Threshold: {min_area_threshold:.2e}")
    print(f"  Max Edge Ratio (sliveriness): {max_edge_ratio:.1f}")

    num_cells = polydata.GetNumberOfCells()

    for cellId in range(num_cells):
        cell = polydata.GetCell(cellId)

        if cell.GetCellType() != vtk.VTK_TRIANGLE or cell.GetNumberOfPoints() != 3:
            continue  # Only interested in valid triangles

        polydata.GetPoint(cell.GetPointId(0), p1)
        polydata.GetPoint(cell.GetPointId(1), p2)
        polydata.GetPoint(cell.GetPointId(2), p3)

        # Check for collapsed points (0-length edges)
        d12_sq = vtk.vtkMath.Distance2BetweenPoints(p1, p2)
        d23_sq = vtk.vtkMath.Distance2BetweenPoints(p2, p3)
        d31_sq = vtk.vtkMath.Distance2BetweenPoints(p3, p1)

        is_degenerate_by_points = False
        if (
            d12_sq < 1e-12 or d23_sq < 1e-12 or d31_sq < 1e-12
        ):  # Very small squared distance
            is_degenerate_by_points = True

        # Check for very small area
        area = vtk.vtkTriangle.TriangleArea(p1, p2, p3)
        is_too_small_area = (
            area < min_area_threshold and not is_degenerate_by_points
        )  # Only flag if not already degenerate by points

        # Check for sliveriness (ratio of longest edge to shortest edge)
        edges = [d12_sq**0.5, d23_sq**0.5, d31_sq**0.5]
        max_edge = max(edges)
        min_edge = min(
            e for e in edges if e > 1e-12
        )  # Avoid division by zero for truly collapsed edges

        is_sliver = False
        if min_edge > 0 and (max_edge / min_edge) > max_edge_ratio:
            is_sliver = True

        if is_degenerate_by_points or is_too_small_area or is_sliver:
            degenerate_cell_ids.InsertNextId(cellId)
            print(f"  Degenerate Cell ID {cellId}:")
            if is_degenerate_by_points:
                print(f"    - Points too close or coincident.")
            if is_too_small_area:
                print(f"    - Area too small: {area:.2e}")
            if is_sliver:
                print(f"    - Sliver: Max edge / Min edge = {max_edge / min_edge:.1f}")
            print(f"    Vertices: {p1}, {p2}, {p3}")
            print("-" * 30)

    if degenerate_cell_ids.GetNumberOfIds() == 0:
        infoDisplay(f"No degenerate or very small triangles found based on thresholds.")
    else:
        infoDisplay(
            f"Found {degenerate_cell_ids.GetNumberOfIds()} degenerate or very small triangles."
        )

        # Create a new model for visualization
        problem_triangles_polydata = vtk.vtkPolyData()
        problem_triangles_polydata.SetPoints(polydata.GetPoints())

        problem_polys = vtk.vtkCellArray()
        for i in range(degenerate_cell_ids.GetNumberOfIds()):
            cellId = degenerate_cell_ids.GetId(i)
            cell = polydata.GetCell(cellId)
            # Add only valid 3-point triangles
            if cell.GetCellType() == vtk.VTK_TRIANGLE and cell.GetNumberOfPoints() == 3:
                tri = vtk.vtkTriangle()
                for j in range(3):
                    tri.GetPointIds().SetId(j, cell.GetPointId(j))
                problem_polys.InsertNextCell(tri)

        problem_triangles_polydata.SetPolys(problem_polys)

        if problem_triangles_polydata.GetNumberOfCells() > 0:
            outputModelNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", modelNode.GetName() + "_DegenerateTriangles"
            )
            outputModelNode.SetAndObservePolyData(problem_triangles_polydata)
            displayNode = outputModelNode.GetDisplayNode()
            if displayNode:
                displayNode.SetColor(1, 0.5, 0)  # Orange color
                displayNode.SetOpacity(0.8)
                displayNode.SetRepresentationToSurface()
                displayNode.SetEdgeVisibility(1)
                displayNode.SetPointVisibility(1)
                displayNode.SetPointSize(3)
            infoDisplay(
                f"Degenerate triangles extracted to '{outputModelNode.GetName()}' (orange)."
            )
        else:
            infoDisplay("No valid degenerate triangle cells found to extract.")

    print("-" * 60)


# --- Usage Example ---
# model_node = slicer.util.getNode('vtkMRMLModelNode1') # !!! Replace with your model node !!!
# if model_node:
#     # Adjust thresholds based on your model's scale and what you consider "degenerate"
#     analyzeDegenerateTriangles(model_node, min_area_threshold=1e-9, max_edge_ratio=100.0)
# else:
#     errorDisplay("Model node not found.")


import vtk
import slicer
from slicer.util import getNode, infoDisplay, errorDisplay
import math  # Import math for isnan if needed, though not directly used in this version


def analyzeMeshConnectivity(
    modelNode, min_cells_threshold=50, min_volume_threshold=1e-2
):
    """
    Analyzes a 3D Slicer model for disconnected components (islands)
    and reports properties of each component.

    Args:
        modelNode (vtkMRMLModelNode): The Slicer Model node to analyze.
        min_cells_threshold (int): Minimum number of cells for a component to be considered "significant".
                                   Components smaller than this might be problematic islands.
        min_volume_threshold (float): Minimum volume for a component to be considered "significant".
                                      Adjust based on your model's scale (e.g., 1e-8 mm^3 for very small objects).
    """
    if not modelNode:
        errorDisplay("Input model node is None.")
        return

    polydata = modelNode.GetPolyData()
    if not polydata:
        errorDisplay(f"Model node '{modelNode.GetName()}' does not contain polydata.")
        return

    infoDisplay(f"Analyzing connectivity for '{modelNode.GetName()}'...")

    # Create the connectivity filter to find all disconnected regions
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(polydata)
    connectivity.SetExtractionModeToAllRegions()  # Extract all disconnected pieces
    connectivity.ColorRegionsOn()  # Assign a scalar to each region (region ID) - useful for display if needed
    connectivity.Update()

    num_regions = connectivity.GetNumberOfExtractedRegions()

    print(f"\n--- Mesh Connectivity Analysis for '{modelNode.GetName()}' ---")
    print(f"Found {num_regions} disconnected regions.")

    if num_regions == 0:
        infoDisplay("No regions found after connectivity analysis.")
        return

    region_info = []  # List to store dictionaries of information for each region

    # Iterate through each identified region to extract and analyze it individually
    for i in range(num_regions):
        # Create a new connectivity filter to extract just this specific region
        extract_region = vtk.vtkPolyDataConnectivityFilter()
        extract_region.SetInputData(polydata)
        extract_region.SetExtractionModeToSpecifiedRegions()
        extract_region.AddSpecifiedRegion(i)  # Extract region 'i'
        extract_region.Update()

        region_polydata = extract_region.GetOutput()

        num_region_cells = region_polydata.GetNumberOfCells()
        num_region_points = region_polydata.GetNumberOfPoints()

        # Calculate bounding box for the region
        bounds = region_polydata.GetBounds()  # [xmin, xmax, ymin, ymax, zmin, zmax]

        # Calculate volume using vtkMassProperties
        volume = 0.0
        # Volume calculation is only meaningful for closed surfaces.
        # It might be 0 for open surfaces, thin sheets, or degenerate regions.
        if num_region_cells > 0:  # Avoid creating MassProperties for empty regions
            massProperties = vtk.vtkMassProperties()
            massProperties.SetInputData(region_polydata)
            massProperties.Update()
            volume = massProperties.GetVolume()
        # --- Check for Closure (New Part) ---
        is_closed = True
        if num_region_cells > 0:  # Only check if there are cells to analyze
            feature_edges = vtk.vtkFeatureEdges()
            feature_edges.SetInputData(region_polydata)
            feature_edges.FeatureEdgesOff()  # Turn off sharp edges
            feature_edges.NonManifoldEdgesOff()  # Turn off non-manifold edges
            feature_edges.BoundaryEdgesOn()  # IMPORTANT: Turn on boundary edges
            feature_edges.ManifoldEdgesOff()  # Turn off manifold edges
            feature_edges.Update()
            if feature_edges.GetOutput().GetNumberOfCells() > 0:
                is_closed = False
        # --- End of New Part ---
        print(f"  Region {i}:")
        print(f"    Cells: {num_region_cells}")
        print(f"    Points: {num_region_points}")
        print(
            f"    Bounds (XYZ): [{bounds[0]:.4f},{bounds[1]:.4f}] [{bounds[2]:.4f},{bounds[3]:.4f}] [{bounds[4]:.4f},{bounds[5]:.4f}]"
        )
        print(
            f"    Volume (if closed): {volume:.6f}"
        )  # Display volume, often 0 for open/thin surfaces
        print(f"    Is Closed: {is_closed}")  # Report closure status

        is_problematic = False
        if num_region_cells < min_cells_threshold:
            is_problematic = True
            print(
                f"    ---> POTENTIALLY PROBLEMATIC (Cells below threshold: {min_cells_threshold})"
            )
        # Only check volume if it's positive and below threshold (to avoid flagging open surfaces with 0 volume)
        if volume > 0 and volume < min_volume_threshold:
            is_problematic = True
            print(
                f"    ---> POTENTIALLY PROBLEMATIC (Volume below threshold: {min_volume_threshold:.2e})"
            )

        # Store info for later use, especially if you want to extract them as new models
        region_info.append(
            {
                "id": i,
                "num_cells": num_region_cells,
                "num_points": num_region_points,
                "bounds": bounds,
                "volume": volume,
                "is_problematic": is_problematic,
                "polydata": region_polydata,  # Store a reference to the extracted polydata
            }
        )
        print("-" * 30)

    # Optional: Create new Slicer Model nodes for problematic regions for visual inspection
    problematic_regions_count = 0
    for region in region_info:
        if region["is_problematic"]:
            problematic_regions_count += 1
            outputModelNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode",
                f"{modelNode.GetName()}_ProblemRegion_{region['id']}",
            )
            outputModelNode.SetAndObservePolyData(
                region["polydata"]
            )  # Set the extracted polydata

            # Set display properties for easier visualization
            outputModelNode.CreateDefaultDisplayNodes()
            displayNode = outputModelNode.GetDisplayNode()
            if displayNode:
                displayNode.SetColor(0.8, 0.4, 0.0)  # Orange-brown color
                displayNode.SetOpacity(0.9)
                # Show surface with wireframe
                displayNode.SetRepresentation(dn.SurfaceRepresentation)
                displayNode.EdgeVisibilityOn()
                # displayNode.SetRepresentationToSurfaceWithEdges()  # doesn't exist
                displayNode.SetPointSize(2)  # Make points visible
                # displayNode.SetPointVisibility(1) # Doesn't exist

    if problematic_regions_count > 0:
        infoDisplay(
            f"Extracted {problematic_regions_count} potentially problematic regions as new models (orange-brown)."
        )
        # slicer.app.layoutManager().fitViewInAllLayouts()  # Zoom to fit new models
    else:
        infoDisplay("No problematic regions found based on current thresholds.")

    print("-" * 60)
    infoDisplay("Mesh connectivity analysis complete.")


# --- How to use this script in 3D Slicer ---

# 1. Load your 3D model into Slicer (this should be the surface model generated from your segmentation).
# 2. Open the Python Interactor (View -> Python Interactor).
# 3. Copy and paste the entire 'analyzeMeshConnectivity' function definition into the interactor.
# 4. Get a reference to your loaded model node:
#    You can select it in the Data module, and its Node ID (e.g., 'vtkMRMLModelNode1') is shown.
#    Then use:
#    my_model_node = slicer.util.getNode('vtkMRMLModelNode1') # Replace with your model's actual ID

# 5. Call the function with your model node and desired thresholds:
# if my_model_node:
#     # Adjust thresholds based on your model's typical size and what you consider "small"
#     # For example, if your typical cells are around 100 cells, 10 might be a good threshold.
#     # For volume, consider your unit (mm^3, cm^3) and typical feature sizes.
#     analyzeMeshConnectivity(my_model_node, min_cells_threshold=10, min_volume_threshold=1e-8)
# else:
#     errorDisplay("Model node not found. Please ensure the model is loaded and you use its correct ID.")


import slicer
import vtk
import numpy as np


def createSamplingGridAroundModel(modelNode, margin=5.0, spacing=10.0):
    """
    Constructs a regular 3D grid of sampling points around a vtkMRMLModelNode.

    The grid covers the bounding box of the model node plus a user-specified margin,
    and the grid spacing is an input parameter. The function returns a vtkPolyData
    object containing the grid points.

    Args:
        modelNode (vtkMRMLModelNode): The model node around which to create the grid.
        margin (float, optional): The margin to add around the model's bounding box in mm. Defaults to 5.0.
        spacing (float, optional): The desired spacing between grid points in mm. Defaults to 10.0.

    Returns:
        vtkPolyData: A vtkPolyData object containing the grid points. Returns None if modelNode is invalid.
    """

    if not modelNode or not isinstance(modelNode, slicer.vtkMRMLModelNode):
        slicer.util.warningDisplay("Invalid modelNode provided.")
        return None

    modelPolyData = modelNode.GetPolyData()
    if not modelPolyData:
        slicer.util.warningDisplay("Model node does not contain valid polydata.")
        return None

    # Get the bounding box of the model in RAS coordinates
    bounds = [0.0] * 6  # xmin, xmax, ymin, ymax, zmin, zmax
    modelPolyData.GetBounds(bounds)

    # Apply the margin to the bounding box
    xmin = bounds[0] - margin
    xmax = bounds[1] + margin
    ymin = bounds[2] - margin
    ymax = bounds[3] + margin
    zmin = bounds[4] - margin
    zmax = bounds[5] + margin

    gridPoints = vtk.vtkPoints()

    x_coords = [x for x in np.arange(xmin, xmax, spacing)]
    y_coords = [y for y in np.arange(ymin, ymax, spacing)]
    z_coords = [z for z in np.arange(zmin, zmax, spacing)]

    # Generate grid points
    for x in x_coords:
        for y in y_coords:
            for z in z_coords:
                gridPoints.InsertNextPoint(x, y, z)

    gridPointsNumpy = np.array(
        tuple([x, y, z] for x in x_coords for y in y_coords for z in z_coords)
    )

    # Create a vtkPolyData object to store the grid points
    gridPolyData = vtk.vtkPolyData()
    gridPolyData.SetPoints(gridPoints)

    # return gridPolyData
    return gridPointsNumpy


import slicer
import vtk
import numpy as np


def testPointsInsideSegment(segmentationNode, segmentName, queryPointsRAS):
    """
    Checks if a list of 3D points (RAS coordinates) are inside a specific segment's
    labelmap representation.

    Args:
        segmentationNode (vtkMRMLSegmentationNode): The segmentation node containing the segment.
        segmentName (str): The name (ID) of the target segment.
        queryPointsRAS (np.array): A Nx3 NumPy array of query points in RAS coordinates.

    Returns:
        np.array: A 1D NumPy boolean array of shape (N,), where True indicates the
                  corresponding query point is inside the segment, and False indicates
                  it's outside (including points outside the labelmap boundaries).
                  Returns None if the segment or segmentation node is not found,
                  or if the labelmap cannot be generated.
    """
    if not segmentationNode:
        print("Error: segmentationNode is None.")
        return None

    segmentation = segmentationNode.GetSegmentation()
    if not segmentation:
        print(
            f"Error: Segmentation object not found in node {segmentationNode.GetName()}."
        )
        return None

    segment = segmentation.GetSegment(segmentName)
    if segment:
        segmentId = segmentName
    else:  # not segment
        print(
            f"Error: Segment '{segmentName}' not found in segmentation node '{segmentationNode.GetName()}'."
        )
        # Try finding by name if ID was used instead of name
        segmentId = segmentation.GetSegmentIdBySegmentName(segmentName)
        if segmentId:
            segment = segmentation.GetSegment(segmentId)
            print(f"Note: Found segment by name, its ID is '{segmentId}'. Using it.")
        else:
            return None

    binaryLabelmap = slicer.vtkOrientedImageData()
    wasCreated = segmentationNode.GetBinaryLabelmapRepresentation(
        segmentId, binaryLabelmap
    )

    if not wasCreated or binaryLabelmap.GetNumberOfPoints() == 0:
        print(
            f"Error: Binary labelmap representation for segment '{segmentName}' is missing or empty."
        )
        print(
            "Please ensure the segment has a binary labelmap representation generated (e.g., by loading a labelmap, or by using CreateBinaryLabelmapRepresentation())."
        )
        return None

    # 2. Get Image Geometry Information directly from the vtkOrientedImageData
    dimensions = binaryLabelmap.GetDimensions()

    # Get the IJK to RAS transform (and its inverse) directly from vtkOrientedImageData
    ijkToRasMatrix = vtk.vtkMatrix4x4()
    binaryLabelmap.GetImageToWorldMatrix(ijkToRasMatrix)  # This gives IJK to RAS

    # We need RAS to IJK for our query points
    rasToIjkMatrix = vtk.vtkMatrix4x4()
    rasToIjkMatrix.DeepCopy(ijkToRasMatrix)
    rasToIjkMatrix.Invert()

    # 3. Prepare Query Points (already a NumPy array)
    numPoints = queryPointsRAS.shape[0]
    is_inside_segment = np.zeros(numPoints, dtype=bool)

    # Convert vtkOrientedImageData to NumPy array for efficient lookup (K, J, I) order
    # Note: We need to convert vtkOrientedImageData to a vtkMRMLVolumeNode temporarily
    # to use slicer.util.arrayFromVolume, or directly use vtkImageData methods.
    # For performance with many points, copying to numpy is generally best.

    # Create a temporary MRML volume node to wrap the vtkOrientedImageData
    # This is a bit of a workaround, as arrayFromVolume expects a MRMLVolumeNode.
    tempVolumeNode = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLabelMapVolumeNode",
        f"Temp_Labelmap_Wrap_{segmentName}_{slicer.app.sessionId()}",
    )
    tempVolumeNode.SetAndObserveImageData(binaryLabelmap)  # Set the image data

    # Now convert to numpy array
    labelmap_np = slicer.util.arrayFromVolume(tempVolumeNode)
    dims_np = labelmap_np.shape  # (depth, height, width) -> (K, J, I)

    # For each query point:
    # Convert RAS to IJK, check bounds, sample labelmap
    for i in range(numPoints):
        ras_point = queryPointsRAS[i, :]

        # Convert RAS to IJK using the matrix
        ijk_float = [0, 0, 0, 1]  # Homogeneous coordinate
        ras_to_transform = [ras_point[0], ras_point[1], ras_point[2], 1]
        rasToIjkMatrix.MultiplyPoint(ras_to_transform, ijk_float)

        # Round to nearest integer voxel coordinates
        # ijk_float[0] corresponds to I, ijk_float[1] to J, ijk_float[2] to K
        x_ijk = int(round(ijk_float[0]))
        y_ijk = int(round(ijk_float[1]))
        z_ijk = int(round(ijk_float[2]))

        # Check if IJK coordinates are within the labelmap's bounds
        # Remember NumPy array indexing is (z, y, x) for (depth, height, width)
        if (
            0 <= z_ijk < dims_np[0]
            and 0 <= y_ijk < dims_np[1]
            and 0 <= x_ijk < dims_np[2]
        ):

            # Get the label value at this voxel
            label_value = labelmap_np[z_ijk, y_ijk, x_ijk]

            # If the label value is 1 (our segment), mark as inside
            if label_value == 1:
                is_inside_segment[i] = True
            # Else (label_value is 0 or other background value), mark as false (already initialized to false)
        else:
            # Point is outside the labelmap's extent, so it's outside the segment
            pass  # is_inside_segment[i] is already False

    # Clean up the temporary volume node
    slicer.mrmlScene.RemoveNode(tempVolumeNode)

    return is_inside_segment


def identifyModelProblems(
    modelNode, segmentationNode, segmentName, gridSpacingMm=5, modelBoundsMarginMm=10
):
    """Use a 3D test grid of points to probe model node to identify
    flawed areas where distances are negative when they shouldn't be.
    """
    gridPoints = createSamplingGridAroundModel(
        modelNode, modelBoundsMarginMm, gridSpacingMm
    )
    insideMask = testPointsInsideSegment(segmentationNode, segmentName, gridPoints)
    outsidePoints = gridPoints[np.logical_not(insideMask)]
    distFcn = getModelDistanceFunction(modelNode)
    dists = np.ones(outsidePoints.shape[0])
    for idx, p in enumerate(outsidePoints):
        dists[idx] = distFcn(p)
    mask = dists < 0
    negDistPts = outsidePoints[mask]
    closeFcn = getClosestPointFunction(modelNode)
    closestPoints = np.ones((negDistPts.shape[0], 3))
    linkMarkups = []
    for idx, p in enumerate(negDistPts):
        cp_result = closeFcn(p)
        closestPoints[idx, :] = cp_result[0]
        linkMarkup = showLink(p, closeFcn)
        linkMarkups.append(linkMarkup)
