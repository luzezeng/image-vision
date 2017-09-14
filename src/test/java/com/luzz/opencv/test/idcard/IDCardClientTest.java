package com.luzz.opencv.test.idcard;

import com.luzz.opencv.idcard.IDCardClient;
import com.luzz.opencv.idcard.impl.Tess4Jer;
import org.opencv.core.Core;

/**
 * Test for IDCardClientTest
 */
public class IDCardClientTest {
    public static void main(String[] args) throws Exception {
        System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
        IDCardClient client = new Tess4Jer();
        System.out.println(client.recognize("/Users/joseph/workspace/image-vision/sourceimages/test.png"));
    }
}
