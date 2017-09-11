package com.luzz.opencv;

import com.luzz.opencv.image.ImageHelper;
import com.luzz.opencv.image.enums.IDCard;
import com.luzz.opencv.image.enums.ImageType;
import com.luzz.opencv.ml.enums.LanguageType;
import com.luzz.opencv.ml.Recognizor;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;
import java.util.*;

/**
 * First Version
 * 缺陷：1. 图片大小受限. 改进：Resize.
 *      2. 灰度化调用openCV库. 改进：后面手动采用R通道获取.
 *      3. 二值化阈值固定. 改进：后面需要对图像计算后动态获取.
 *      4. 对字符的识别调用google开源库. 改进：后面需要通过opencv-ml库自定义机器学习算法训练生成策略.
 *
 * 后面计划：学习书《学习OpenCV(中文版)》
 */
public class OpencvTess4J {
    private static final String TEMP_IMAGE_STORAGE_PATH = "/Users/joseph/workspace/IDRecognize/tempimages/";
    private static final String NUMBER_PATH_IMG_NAME = "numberPart.jpg";

    public static String recognize(String imgPath) throws Exception {
        //gray
        Mat grayMat = ImageHelper.generateGray(Imgcodecs.imread(imgPath));
        ImageHelper.generateGrayImageFile(grayMat, TEMP_IMAGE_STORAGE_PATH + "gray.jpg", ImageType.JPG);

        //binary
        Mat binaryMat = ImageHelper.generateNormalBinary(grayMat);
        ImageHelper.generateGrayImageFile(binaryMat, TEMP_IMAGE_STORAGE_PATH + "binary.jpg", ImageType.JPG);

        //get ID card attributes
        Map<IDCard, RotatedRect> rotatedRectMap = ImageHelper.findIDCardAttribute(ImageHelper.detectContours(grayMat, binaryMat));

        //get number part
        Mat IdArea = ImageHelper.cutArea(binaryMat, rotatedRectMap.get(IDCard.ID));
        ImageHelper.generateGrayImageFile(IdArea, TEMP_IMAGE_STORAGE_PATH + "numberPart.jpg", ImageType.JPG);

        //recognize the picture
        return Recognizor.recognizeByTess4J(new File(TEMP_IMAGE_STORAGE_PATH + NUMBER_PATH_IMG_NAME), LanguageType.ENGLIST);
    }
}
