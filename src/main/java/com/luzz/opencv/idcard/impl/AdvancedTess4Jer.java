package com.luzz.opencv.idcard.impl;

import com.luzz.opencv.idcard.IDCardClient;
import com.luzz.opencv.idcard.image.ImageHelper;
import com.luzz.opencv.idcard.image.enums.IDCard;
import com.luzz.opencv.idcard.image.enums.ImageType;
import com.luzz.opencv.idcard.ml.enums.LanguageType;
import com.luzz.opencv.idcard.ml.Recognizor;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import java.io.File;
import java.util.*;

/**
 * First Version
 *
 * Disadvantage：1. Recognition for Chinese characters hasn't worked yet
 *              2. Words' recognition is based on Tess4j. We will going to replace that from ml algorithm
 *
 * Later Plan：Learn Machine Learning and OpenCV
 *
 * @author Joseph Lu
 */
public class AdvancedTess4Jer implements IDCardClient {
    private static final String TEMP_IMAGE_STORAGE_PATH = "/Users/joseph/workspace/IDRecognize/tempimages/";
    private static final String NUMBER_PATH_IMG_NAME = "numberPart.jpg";

    public Map<String, String> recognize(String imgPath) throws Exception {
        //gray from R channel
        Mat grayMat = ImageHelper.obtainRChannel(Imgcodecs.imread(imgPath));
        ImageHelper.generateGrayImageFile(grayMat, TEMP_IMAGE_STORAGE_PATH + "gray.jpg", ImageType.JPG);

        //binary from self-adaption
        Mat binaryMat = ImageHelper.generateSelfAdaptionBinary(grayMat);
        ImageHelper.generateGrayImageFile(binaryMat, TEMP_IMAGE_STORAGE_PATH + "binary.jpg", ImageType.JPG);

        //get ID card attributes
        Map<IDCard, RotatedRect> rotatedRectMap = ImageHelper.findIDCardAttribute(ImageHelper.detectContours(grayMat, binaryMat));

        //get number part
        Mat IdArea = ImageHelper.cutArea(binaryMat, rotatedRectMap.get(IDCard.ID));
        ImageHelper.generateGrayImageFile(IdArea, TEMP_IMAGE_STORAGE_PATH + "numberPart.jpg", ImageType.JPG);

        //recognize the picture
        final String id = Recognizor.recognizeByTess4J(new File(TEMP_IMAGE_STORAGE_PATH + NUMBER_PATH_IMG_NAME), LanguageType.ENGLIST);

        return new HashMap<String, String>() {
            {
                put("ID", id);
            }
        };
    }
}