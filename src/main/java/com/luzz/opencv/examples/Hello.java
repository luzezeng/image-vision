package com.luzz.opencv.examples;

import com.luzz.opencv.OpencvTess4J;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * @author Joseph
 *
 * an example to show how to user opencv library.
 */
public class Hello {
    public static void main( String[] args ) throws Exception {
        System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
        Mat mat = Mat.eye( 3, 3, CvType.CV_8UC1 );
        System.out.println( "mat = " + mat.dump() );
        System.out.println(OpencvTess4J.recognize("/Users/joseph/workspace/IDRecognize/images/02.png"));
    }
}
