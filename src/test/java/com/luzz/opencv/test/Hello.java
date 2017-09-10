package com.luzz.opencv.test;

import com.luzz.opencv.OpencvTess4J;
import org.opencv.core.Core;

/**
 * @author Joseph
 *
 * an example to show how to user opencv library.
 */
public class Hello {
    public static void main( String[] args ) throws Exception {
        System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
        System.out.println(OpencvTess4J.recognize("/Users/joseph/workspace/IDRecognize/sourceimages/test.png"));
    }
}
