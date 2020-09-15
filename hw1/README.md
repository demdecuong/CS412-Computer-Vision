

## Installation
- cv2
- math
- numpy
## Usage
`python main.py example.png`

**Supported features**  

    "[i] - reload the original image (i.e. cancel any previous processing)",
    "[w] - save the current (possibly processed) image into the file out.jpg",
    "[g] - convert the image to grayscale using the openCV conversion function.",
    "[q] - convert the image to grayscale using your implementation of conversion function.",
    "[c] - cycle through the color channels of the image.",
    "[s] - convert the image to grayscale and smooth it using the openCV function.",
    "[d] - convert the image to grayscale and smooth it using my function.",
    "[x] - convert the image to grayscale and perform convolution with an x derivative filter.",
    "[y] - convert the image to grayscale and perform convolution with a y derivative filter.",
    "[m] - show the magnitude of the gradient normalized to the range [0,255].",
    "[v] - convert the image to grayscale and plot the gradient vectors of the image every N pixels.",
    "[r] - convert the image to grayscale and rotate it using an angle of Q degrees.",
    "[h] - Display this help window.",
    "[p] - Quit the program."