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
        System.out.println(OpencvTess4J.recognize("/Users/joseph/workspace/IDRecognize/images/02.png"));
    }
}
