---
layout: post
title:  "A few notes on using the Tensorflow C++ API"
date:   2016-06-10 22:10:33 +0200
permalink: deeplearning/tensorflow-cpp

---

If you are unfamiliar with bazel, then there are some quirks in getting TensorFlow to work with OpenCV, optimizations turned on, and with building shared libraries.

Creating a binary compiled against Tensorflow with bazel
---------------------------------------------------
 - Clone the tensorflow repository.
 - Inside tensorflow/tensorflow, create a working directory.
 - Add your C++ code that uses tensorflow, lets put that in code.cpp.
 - Create a file called BUILD that looks like this:
 {% highlight bash %}
 cc_binary(
    name = "project",
    srcs = ["code.cpp"],
    linkopts = 
		["-lopencv_core","-lopencv_imgproc", "-lopencv_highgui", 
		"-Wl,--version-script=tensorflow/tf_version_script.lds"],
    copts = ["-I/usr/local/include/", "-O3"],
    deps = [
        "//tensorflow/core:tensorflow"
    ])
{% endhighlight %}	        


 - To build (with optimizations turned on): `bazel build -c opt :project`. 
The binary will be in bazel-bin/tensorflow/project. 


Creating a shared library compiled against Tensorflow with bazel
=======
Here we want to build a shared library with C++ code that uses the Tensorflow C++ API.
This will probably be the common case for production use, since you will have a large code base with its own build system (like CMake), but you need to call Tensorflow.
Building against Tensorflow restricts you to bazel (at least that seems the simplest way for now), but you can create a shared library that can be called from the larger code base.

The main issue is that bazel outputs a shared library containing only Tensorflow symbols (checked with nm -g), and *.o object files with the C++ files compiled.
That is kind of weird behaviour and seems to be an issue with bazel.
We will deal with that by just compiling against both files (the actual dynamic loaded shared library will have the tensorflow part, and our C++ client code in the object file will be linked statically. You can also just wrap the object file in another shared library).

The BUILD file should now look like this:

{% highlight bazel %}
`cc_binary(
    name = "libproject.so",
    srcs = ["code.cpp"],
	linkopts = ["-shared", "-lopencv_core","-lopencv_imgproc", "-lopencv_highgui", "-Wl,--version-script=tensorflow/tf_version_script.lds"],
	linkshared=1,
	copts = ["-I/usr/local/include/", "-O3"],
    deps = [
        "//tensorflow/core:tensorflow"
    ],
    visibility=["//visibility:public"]`
{% endhighlight %}
- To build: `bazel build -c opt --copt="-fPIC" :libproject.so` 
- Now in bazel-bin we will have both libproject_name.so (containing only Tensorflow symbols), and the object files with our client code (under _objs/libproject.so/tensorflow/code/code.o.

Now you can compile against both files, for example like this:

{% highlight bash %}
g++ -o main main.cc tensorflow/bazel-bin/tensorflow/project/_objs/libproject.so/tensorflow/project/code.o pkg-config --libs opencv -lproject -L tensorflow/bazel-bin/tensorflow/project
{% endhighlight %}