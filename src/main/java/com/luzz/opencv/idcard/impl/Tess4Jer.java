package com.luzz.opencv.idcard.impl;

import com.luzz.opencv.idcard.IDCardClient;
import com.luzz.opencv.idcard.handler.image.ImageHandler;
import com.luzz.opencv.idcard.handler.image.enums.IDCard;
import com.luzz.opencv.idcard.handler.image.enums.ImageType;
import com.luzz.opencv.idcard.handler.ml.enums.LanguageType;
import com.luzz.opencv.idcard.handler.ml.MLHandler;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.util.*;

/**
 * First Version
 * Disadvantage：1. The size of the picture is an effective factor
 *              2. The Gray Scala's generation is all dependent on the OpenCV tool. We'll make it use the R channel by hand.
 *              3. The limit of binaryzation is fixed. We should get the value varied in different picture.
 *              4. Still based on Tess4j. We should code the ml algorithm.
 *
 * Later plan：Learn OpenCV
 *
 * @author Joseph Lu
 */
public class Tess4Jer implements IDCardClient {
    private static final String TEMP_IMAGE_STORAGE_PATH = "/Users/joseph/workspace/image-vision/tempimages/";
    private static final String NUMBER_PATH_IMG_NAME = "numberPart.jpg";

    public Map<String, String> recognize(String imgPath) throws Exception {
        //gray
        Mat grayMat = ImageHandler.generateGray(Imgcodecs.imread(imgPath));
        ImageHandler.generateGrayImageFile(grayMat, TEMP_IMAGE_STORAGE_PATH + "gray.jpg", ImageType.JPG);

        //binary
        Mat binaryMat = ImageHandler.generateNormalBinary(grayMat);
        ImageHandler.generateGrayImageFile(binaryMat, TEMP_IMAGE_STORAGE_PATH + "binary.jpg", ImageType.JPG);

        //get ID card attributes
        Map<IDCard, RotatedRect> rotatedRectMap = ImageHandler.findIDCardAttribute(ImageHandler.detectContours(grayMat, binaryMat));

        //get number part
        Mat IdArea = ImageHandler.cutArea(binaryMat, rotatedRectMap.get(IDCard.ID));
        ImageHandler.generateGrayImageFile(IdArea, TEMP_IMAGE_STORAGE_PATH + "numberPart.jpg", ImageType.JPG);

        //recognize the picture
        final String id = MLHandler.recognizeByTess4J(new File(TEMP_IMAGE_STORAGE_PATH + NUMBER_PATH_IMG_NAME), LanguageType.ENGLIST);

        return new HashMap<String, String>() {
            {
                put("ID", id);
            }
        };
    }
}
