A program written for a machine learning class, to sort in-focus photos from out-of-focus photos.

For a standard photo, where most of the photo is entirely in-focus or out-of-focus, this could just be done using the laplacian variance, however, my goal was to build a more sophisticated program that could tell the difference for event photos in low light, where often only parts of the subject are in focus, usually (preferably) the subject's face.

I was curious to see if I could build a machine learning program that would learn to tell if a photo was in focus or not, using as data a series of edge and focus measurements from a variety of regions around the photo (using face-recognition algorithms, or just considering the central third of the photo, etc.)

I manually constructed some training and test data from several events I've photographed in the past. Much of the code itself is a test harness to try various different machine learning algorithms. The most effective algorithm found was Random Forest, which correctly identified about 86.4% of the test set.

I'd like to do some additional refinement on this program in the future. If I could get it more reliably working, it would save a lot of time after photography gigs!