### Pento Pal

Below is a picture of the classic [6x10 pentomino puzzle](https://en.wikipedia.org/wiki/Pentomino#Constructing_rectangular_dimensions). There are 12 different pieces, one for each way to connect 5 squares into one shape.

<p align="center">
    <img alt="Original Image" src="docs/step-0-original-image.png">
</p>

If you can believe it, there are 2,339 unique solutions to this puzzle! As one finds more and more of these solutions, it becomes difficult to track whether you've found a new solution or simply stumbled across a previously seen one.

Pento Pal is an app for tracking the solutions you've found and identifying if a new solution is actually new or just a repeat. It uses an ML pipeline to parse a picture of the puzzle into a description of the piece states. Specifically, the input is an image of a solved puzzle and the output is a 2-d array indicating which piece is present in each cell in the 6x10 puzzle grid.

### How it works

* Step 1: Run the image through a pose estimation model fine-tuned to predict keypoints representing the corners of the puzzle.
* Step 2: Using the corner points from the previous step, apply a perspective transformation to align the puzzle.
* Step 3: Run the image through an object detection model fine-tuned to predict bounding boxes and class (piece type) for all of the pieces.
* Step 4: Take the bounding box for the entire puzzle and divide evenly into a 6x10 grid.
* Step 5: Take the unaligned bounding boxes from step 3 and "snap" them to the grid from step 4. E.g. Map them from pixel coordinates to "grid" coordinates.
* Step 6: We now know where the pieces are in the puzzle grid, but not their orientation. Fortunately, if the image is of a valid solution and there were no errors in the previous steps, then there is only one way to orient the 12 pieces without overlaps. So we can simply iterate over possible orientations until we find the correct set.

Here are visualizations of the first 5 steps:

<p float="left">
    <img alt="Step 1" src="docs/step-1.png">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Step 2" src="docs/step-2.png">
</p>

<p float="left">
    <img alt="Step 3" src="docs/step-3.png">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Step 4" src="docs/step-4.png">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Step 5" src="docs/step-5.png">
</p>
