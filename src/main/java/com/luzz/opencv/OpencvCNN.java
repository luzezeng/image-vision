package com.luzz.opencv;

import com.luzz.opencv.image.ImageHelper;
import com.luzz.opencv.image.enums.IDCard;
import com.luzz.opencv.image.enums.ImageType;
import com.luzz.opencv.ml.Recognizor;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.*;

/**
 * Second Version
 *
 * Some tips:
 *      1. Have been improved in every item mentioned in the last version
 *      2. Still the annconfig.xml file's generating and reading has not finished
 *      3. Will add other character's recognization
 *      4. Will improve the accuracy
 *      5. Will try other machine learning method
 * 后面计划：学习书《学习OpenCV(中文版)》
 *
 * @author Joseph Lu
 */
public class OpencvCNN {
    private static final String TEMP_IMAGE_STORAGE_PATH = "/Users/joseph/workspace/IDRecognize/tempimages/";

    public static String recognize(String imgPath) throws Exception {
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

        //Cut the number mat
        List<Mat> charMat = ImageHelper.cutChar(IdArea);
        for (int i = 0; i < charMat.size(); i ++) {
            ImageHelper.generateGrayImageFile(charMat.get(i), TEMP_IMAGE_STORAGE_PATH + "/chars/" + i + ".jpg", ImageType.JPG);
        }

        return String.valueOf(Recognizor.recognizeIDCardByCNN(charMat));
    }
}
