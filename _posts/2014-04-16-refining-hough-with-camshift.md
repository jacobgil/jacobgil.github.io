---
layout: post
title:  "Refining the Hough Transform with CAMSHIFT"
date:   2014-05-05 22:10:33 +0200
categories: MachineLearning
---

The Circular Hough Transform result is often not very accurate due to noise\details\occlusions.  

Typical ways of dealing with this are:  
1\. Hand tuning the Hough Transform parameters.  
2\. Pre-processing the image aggressively before the transform is applied.  

One trick I use to fix the circles positions is an iterative search in windows around the initial circles, I hope to have a future post about this here.  

But now I will share a much simpler strategy that works well in some cases: Use CAMSHIFT to track the circular object in a window around the initial circles positions.  

The idea is that the initial circle center position area holds information about how the circular object looks like, for example its color distribution. This is complementary to the Hough transform that uses only spatial information (the binary votes in the Hough space).  

### The steps
1.  Find circles with the Circular Hough Transform.
2.  Find the histogram inside a small box around each circle. <span style="text-align: center;">In a more general case we can use any kind of features we like, like texture features or something, </span>but here we will stick with color features.
3.  For each pixel, find the probability it belongs to the circular object (back-projection).
4.  Optional: apply some strategy to fill holes in the back-projection image caused by occlusions. We can use morphology operations like dilating for example.
5.  Use CAMSHIFT to track the the circular object starting in a window around the initial circle position.

[![](http://2.bp.blogspot.com/-c9yGeYXMjGs/U047G2o5cBI/AAAAAAAAFgQ/GDTWC5m1u7Q/s1600/initial.jpg)](http://2.bp.blogspot.com/-c9yGeYXMjGs/U047G2o5cBI/AAAAAAAAFgQ/GDTWC5m1u7Q/s1600/initial.jpg)  [![](http://1.bp.blogspot.com/-iarLgzZ5jIk/U04609-Q0QI/AAAAAAAAFgE/1kGJ_Lcy7y4/s1600/refined.jpg)](http://1.bp.blogspot.com/-iarLgzZ5jIk/U04609-Q0QI/AAAAAAAAFgE/1kGJ_Lcy7y4/s1600/refined.jpg)

Conveniently for us, CAMSHIFT is included in OpenCV!  

**I encourage you to read the original CAMSHIFT paper to learn more about it:**  
**_Computer Vision Face Tracking For Use in a Perceptual User_**  
**_Interface_**  
**_Gary R. Bradski, Microcomputer Research Lab, Santa Clara, CA, Intel Corporation_**  
[Link to the paper](http://www.cse.psu.edu/~rcollins/CSE598G/papers/camshift.pdf)  

Code (C++, using OpenCV):

{% highlight c++ %}
    #include "opencv2/highgui/highgui.hpp"
    #include "opencv2/imgproc/imgproc.hpp"
    #include "opencv2/core/core.hpp"
    #include "opencv2/video/tracking.hpp"
    #include <tuple>

    using namespace cv;
    using namespace std;

    //This is used to obtain a window inside the image,
    //and cut it around the image borders.
    Rect GetROI(Size const imageSize, Point const center, int windowSize)
    {
        Point topLeft (center.x - windowSize / 2, center.y - windowSize / 2);

         if (topLeft.x + windowSize > imageSize.width || 
             topLeft.y + windowSize >
             
             imageSize.height)
        {
             windowSize = 
             min(imageSize.width - topLeft.x, imageSize.height - topLeft.y);
        } 
        return Rect(topLeft.x, topLeft.y, windowSize, windowSize);
    }

    // This is used to find pixels that likely belong to the circular object
    // we wish to track.
    Mat HistogramBackProjectionForTracking(Mat const& image, Rect const window)
    {
         const int sizes[] = {256,256,256};
         float rRange[] = {0,255};
         float gRange[] = {0,255};
         float bRange[] = {0,255};
         const float *ranges[] = {rRange,gRange,bRange};
         const int channels[] = {0, 1, 2};

         Mat roi = image(window);

         Mat hist;
         if (image.channels() == 3)
          calcHist(&roi, 1, channels, Mat(), hist, 3, sizes, ranges);
         else
          calcHist(&roi, 1, &channels[0], Mat(), hist, 1, &sizes[0], 
            &ranges[0]);

         Mat backproj;
         calcBackProject(&image, 1, channels, hist, backproj, ranges);
         return backproj;
    }

    // Return a new circle by using CAMSHIFT 
    // to track the object around the initial circle.
    tuple<Point, int> 
    HoughShift(Point const center, int const radius, Mat const& image)
    {
        Mat backproj = HistogramBackProjectionForTracking(image, 
        GetROI(image.size(),center, radius));

         //Fill holes:
         cv::dilate(backproj, backproj, cv::Mat(), cv::Point(-1,-1));
         cv::dilate(backproj, backproj, cv::Mat(), cv::Point(-1,-1));

        const int windowTrackingSize = 4 * radius;
        RotatedRect track = CamShift(backproj, GetROI(image.size(), center,
        TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

        return make_tuple(track.center, 
                         (track.size.width + track.size.height )/ 4);
    }

    int main(int argc, char** argv)
    {
         Mat image = cv::imread("image.jpg");

         Mat before, after; image.copyTo(before); image.copyTo(after);
         Mat gray; cv::cvtColor(image, gray, CV_BGR2GRAY);

         std::vector<cv::Vec3f> circles;
         HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 2, 
            gray.cols / 3, 20, 40,
         gray.cols / 20, gray.cols / 5);

         for (int i = 0; i < circles.size(); ++i)
         {
             auto circle = HoughShift(Point(circles[i][0], circles[i][1]), 
             circles[i][2], image);

             circle(before, 
                Point(circles[i][0], circles[i][1]), circles[i][2], 
             Scalar(128, 128, 30),  2);
             circle(after, get<0>(circle), get<1>(circle), 
                Scalar(255, 0 , 0), 2);
         }

         imshow("Initial Circles", before);
         imshow("Refined Circles", after);
         waitKey(-1);

         return 0;
    }
{% endhighlight %}