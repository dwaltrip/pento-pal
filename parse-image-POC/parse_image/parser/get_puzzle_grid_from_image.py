
from settings import GRID
from parse_image.parser.engine import (
    get_puzzle_box_corners,
    dewarp_rectangle,
    get_piece_bounding_boxes,
    # simple_map_boxes_to_grid,
)
from parse_image.parser.bounding_boxes_to_grid_boxes import (
    bounding_boxes_to_grid_boxes,
)
from parse_image.parser.piece_grid_boxes_to_puzzle_grid import (
    get_puzzle_grid_from_piece_boxes
)


def get_puzzle_grid_from_image(image):
    """
    End-to-end solution for extracting the puzzle grid from an image:

        1. Use pose-estimation model to get keypoints for corners and the
            dimensions of the "puzzle box" in the image.
        2. Use keypoints to normalize the image of the puzzle (so the puzzle looks
            perfectly straight and aligned)
        3. Use "piece detection" (obj detect) model to detect the bounding boxes of each piece.
        4. Using the results from step 1, evenly divide the "puzzle box" region of the image,
            which is now a proper rectangle, into a grid.
        5. Match the bounding box for each piece to grid cells from step 4.
            For borderline cases, take the grid cells that overlap more.
            I think that should work 99% of the time.
        6. Sort the pieces by their top left grid cell, and run the algo that simply tries
            each orientation for each piece, starting from top-left and working down to
            bottom-right. This is already implemented in `get_puzzle_grid_from_piece_boxes`. 
        
    And voila, now we have the solution in the image represented as 2d-array of class names (pieces)!
        We did it :)
    """

    # Step 1
    puzzle_corners = get_puzzle_box_corners(image)

    # ------------------------------------------------------------------------
    # TODO: stupid hack due to the stupid corner ordering issue...
    pc = puzzle_corners
    # 3rd and 4th corner, bottom left and bottom right, are in wrong order
    puzzle_corners = (pc[0], pc[1], pc[3], pc[2]) 
    # ------------------------------------------------------------------------

    # Step 2
    aspect_ratio = GRID.width / GRID.height
    normalized_image = dewarp_rectangle(image, puzzle_corners, aspect_ratio)

    # TODO: for funzies draw the grid on it.
    # normalized_image.show()

    # Step 3
    piece_bounding_boxes = get_piece_bounding_boxes(normalized_image)

    # Step 4 and 5
    # piece_grid_boxes = simple_map_boxes_to_grid(piece_bounding_boxes)
    piece_grid_boxes = bounding_boxes_to_grid_boxes(piece_bounding_boxes)

    # Step 6
    puzzle_grid = get_puzzle_grid_from_piece_boxes(piece_grid_boxes)

    return puzzle_grid

