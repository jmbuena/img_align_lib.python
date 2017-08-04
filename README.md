A python reimplementation of the C++ image alignment library

Test library now:
=================

Pull code from the repository and run:

    $ cd img_alig_lib.python/code
    $ python inv_comp.py --video_source='../videos/book1.mp4' --template_corners 106 166 106 69 211 69 211 166 --template_dims 100 100


Incremental Image Alignment Library
===================================

This is a library of incremental image alingment algorithms using python 2.7 and OpenCV 2.4.X

Incremental Image alingment is used in tracking applications. In this applications you find the
pose of the object in the scene by any means (i.e. object detection) and subsquently the pose
is reestimated in the current frame using the pose in the former frame. In order to
estimate motion, a model of the displacement of the object is assumed. Therefore, in this context
pose stands for the motion parameters.

The elements of any incremental image alignment are:

  * The object model (a single texture image, a planar appearance model, AAM, 3D morphable model, etc).
  * The motion model (2D similarity, 2D affine, 2D homography, 3D pose (rotation and traslation), etc).
  * The optimisation procedure (Gauss-Newton, Levenberg–Marquardt, etc).

Those elements are provided in the library and also an Object Oriented Design that allows to easily introduce
efficient technicques for incremental image alignment:

  * Hager and Belhumeur's jacobian factorisation method::
      
        Efficient Region Tracking with Parametric Models of Geometry and Illumination.
        G. Hager, P. Belhumeur
        IEEE Trans. PAMI, 20(10), pp. 1025-39, October 1998.

  * Baker and Matthews's Inverse Compositional method::
      
        Lucas-Kanade 20 Years On: A Unifying Framework.
        Simon Baker, Iain Matthews
        International Journal of Computer Vision 56(3), 221–255, 2004

The img_aling_lib.python already provides:

  * Image Compositional for planar tracking (with a single texture object model) with
    homography based motion model.
  * A demo script for inverse compositional planar tracking.

Requirements
------------

You need the following libraries installed on you system in order to
build img_align_lib.python:

* `OpenCV <http://www.opencv.org/>`
* python 2.7

