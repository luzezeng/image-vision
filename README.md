# Welcome to Image-vision's Codebase, owned by Joseph Lu

Program language is Java

Current Build Status: ![Build Status](https://travis-ci.org/bitcoin-dot-org/bitcoin.org.svg?branch=master)

Report problems or help improve the site by opening a [new issue](https://github.com/luzezeng/image-vision/issues/new) or [pull request](https://github.com/luzezeng/image-vision/compare).

## What I have done
#### 1. Recognize numbers or words in any identity card from Chinese based on OpenCV and Tess4J
    Handle the gray scale and binaryzation issue by tools provided in OpenCV, and recognize the numbers or words with the help of Tess4J.

#### 2. Recognize numbers in any identity identity card from Chinese based on OpenCV and machine learning algorithm
    Instead of the Tess4J, the recognition task has been completed by machine learning coding myself, and you can see all the code directly.

## What will be upgraded afterwards
#### 1. Recognize Chinese characters in any identity card and normal bank card based Tess4J

#### 2. Recognize All information in any card

#### 3. Recognize and calculate GPA in a printed transcript

## How to Participate
You can run the project easily in every popular operating system. Take the example of OS, the following quick guides will help you get started:

#### 1. Install OpenCV
    run brew install opencv in terminal

#### 2. Install Tesseract if your running code based on Tess4J
    run brew install tesseract --all-language

#### 3. Change the path of your own directory in the test code

#### 4. Run the project in your own IDE and set the VM parameters
    Take the example of Intellij Idea, you should set the java.library.path formed like -Djava.library.path=/usr/local/opt/opencv3/share/OpenCV/java

#### 4. Tip need to know before
    Be sure that you have install HomeBrew

### Questions?
Please contact Will Joseph Lu ([luzezeng@gmail.com](mailto:luzezeng@gmail.com)) if you need help.
